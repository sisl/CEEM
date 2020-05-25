#
# File: smoother.py
#

import numpy as np
import torch
from numpy.random import choice
from scipy.optimize import least_squares, minimize
from torch.distributions.multivariate_normal import MultivariateNormal

from ceem import utils
from ceem.opt_criteria import GroupSOSCriterion, STRStateCriterion
from tqdm import tqdm


def NLSsmoother(x0, criterion, system, solver_kwargs={'verbose': 2}):
    """
    Smoothing with Gauss-Newton based approach
    Args:
        x0 (torch.tensor): (1,T,n) system states
        criterion (SOSCriterion): criterion to optimize
        system (DiscreteDynamicalSystem)
        solver_kwargs : options for scipy.optimize.least_squares
    """

    if 'tr_rho' in solver_kwargs:
        tr_rho = solver_kwargs.pop('tr_rho')
        criterion = GroupSOSCriterion([criterion, STRStateCriterion(tr_rho, x0)])

    B, T, xdim = x0.shape
    assert B == 1, f"Smoothing one trajectory at a time. x0.shape[0] is {B} but should be 1."

    def loss(x):
        with torch.no_grad():
            x = torch.tensor(x).view(1, T, xdim).to(x0.dtype)
            loss = criterion.residuals(system, x)
        return loss.numpy()

    def jac(x):
        x = torch.tensor(x).view(1, T, xdim)
        return criterion.jac_resid_x(system, x, sparse=True)

    with utils.Timer() as time:
        kwargs = dict(method='trf', loss='linear')
        kwargs.update(solver_kwargs)
        opt_result = least_squares(loss, x0.view(-1).detach().numpy(), jac, **kwargs)

    x = torch.tensor(opt_result.x).view(1, T, xdim)

    metrics = {'fun': float(opt_result.cost), 'success': opt_result.success, 'time': time.dt}

    return x, metrics


def EKF(x0, y_T, u_T, sigma0, Q, R, system):
    """
    Extended Kalman filter

    Args:
        x0 (torch.tensor): (B, xdim) initial system states
        y_T (torch.tensor): (B, T, ydim) observations
        u_T (torch.tensor): (B, T, udim) controls
        sigma0 (torch.tensor): (xdim, xdim) initial state covariance
        dyn_err (torch.tensor): (xdim) dynamics error mean
        obs_err (torch.tensor): (xdim) observation error mean
        Q (torch.tensor): (xdim, xdim) dynamics error covariance
        R (torch.tensor): (ydim, ydim) observation error covariance
        system (DiscreteDynamicalSystem)

    Returns:
        x_filt (torch.tensor): (B, T, xdim) system states
        y_pred (torch.tensor): (B, T, ydim) predicted observations before state correction
    """

    xdim = Q.shape[0]
    B, T, ydim = y_T.shape
    I = torch.eye(xdim)

    x = torch.zeros(B, T, xdim)
    x[:, 0:1] = x0

    y = y_T.clone()
    with torch.no_grad():
        y[:, 0:1] = system.observe(0, x[:, :1], u_T[:, :1])

    St = torch.zeros(B, T, xdim, xdim)
    St[:, 0] = sigma0.unsqueeze(0)

    for t in tqdm(range(1, T)):
        # Propagate dynamics
        with torch.no_grad():
            x[:, t:t + 1] = system.step(t - 1, x[:, t - 1:t], u_T[:, t - 1:t])
        Gt = system.jac_step_x(t, x[:, t:t + 1], u_T[:, t:t + 1]).detach()
        St_hat = Gt @ St[:, t - 1:t] @ Gt.transpose(-1, -2) + Q

        # Estimate observation
        with torch.no_grad():
            y[:, t:t + 1] = system.observe(t, x[:, t:t + 1], u_T[:, t:t + 1])
        Ht = system.jac_obs_x(t, x[:, t:t + 1], u_T[:, t:t + 1]).detach()
        Zt = Ht @ St_hat @ Ht.transpose(-1, -2) + R

        # Estimate Kalman Gain and correct xt
        Kt = St_hat @ Ht.transpose(-1, -2) @ torch.inverse(Zt)
        x[:, t:t + 1] = x[:, t:t + 1] + (Kt @ (
            y_T[:, t:t + 1] - y[:, t:t + 1]).unsqueeze(-1)).squeeze(-1)
        St[:, t:t + 1] = (I - Kt @ Ht) @ St_hat
    return x, y

## Particle Smoother
class ParticleSmootherSystemWrapper:

    def __init__(self, sys, R):
        self._sys = sys
        self._Rmvn = MultivariateNormal(torch.zeros(R.shape[0],), R)

    def __call__(self, x, t):
        """
        t (int): time
        x (torch.tensor): (N,n) particles
        """

        x = x.unsqueeze(1)
        t = torch.tensor([float(t)])
        nx = self._sys.step(t, x)
        return nx.squeeze(1)

    def obsll(self, x, y):
        """
        x (torch.tensor): (N,n) particles
        y (torch.tensor): (1,m) observation
        """
        y_ = self._sys.observe(None, x.unsqueeze(1)).squeeze(1)
        dy = y - y_
        logprob_y = self._Rmvn.log_prob(dy).unsqueeze(1)
        return logprob_y

    @property
    def _xdim(self):
        return self._sys.xdim


class ParticleSmoother:

    def __init__(self, N, system, obsll, Q, Px0, x0mean=None):
        """
        Args:
          N (int):
          system (DiscreteDynamics):
          obsfun (callable): function mapping ((*,xdim),(1,ydim) -> (*,[0,1])
          Q (torch.tensor): (xdim,xdim) torch tensor
          R (torch.tensor): (ydim,ydim) torch tensor
          Px0 (torch.tensor): (xdim,xdim) torch tensor
          x0mean (torch.tensor): (1,xdim) torch tensor

        """

        self._N = N
        self._system = system
        self._obsll = obsll
        self._xdim = system._xdim
        self._Qchol = Q.cholesky().unsqueeze(0)
        self._Qpdf = MultivariateNormal(torch.zeros(self._xdim), Q)
        self._Px0chol = Px0.cholesky().unsqueeze(0)
        if x0mean is not None:
            self._x0mean = x0mean
        else:
            self._x0mean = torch.zeros(1, self._xdim)

        self._xfilt = None
        self._wfilt = None
        self._wsmooth = None


    def filter(self, y):
        # inputs:
        #   y (torch.tensor): (T, ydim) torch tensor

        T = y.shape[0]
        x = torch.zeros(T, self._N, self._xdim)
        w = torch.zeros(T, self._N, 1)

        # sample initial distribution
        x[0] = self._x0mean + (self._Px0chol @ torch.randn(self._N, self._xdim, 1)).squeeze(2)

        for t in range(T - 1):

            ## Observe
            log_wt = self._obsll(x[t], y[None,t])
            
            ## Update weights
            # numerically stable computation of w
            log_wt -= log_wt.max()
            wt = log_wt.exp()
            wt /= wt.sum()
            # since we divide by wt.sum(), subtracting off log_wt.max()
            # gives the same result
            w[t] = wt

            ## Resample
            rinds = choice(self._N, self._N, p=w[t, :, 0].detach().numpy())
            xtr = x[t,rinds]
            
            ## Propegate
            with torch.no_grad():
                x[t + 1] = self._system(
                    xtr, t) + (self._Qchol @ torch.randn(self._N, self._xdim, 1)).squeeze(2)

        log_wt = self._obsll(x[-1], y[None, -1])
        log_wt -= log_wt.max()
        wt = log_wt.exp()
        wt /= wt.sum()
        w[-1] = wt

        return x, w

    def smoother(self, x, w):

        T, N, n = x.shape

        ## Compute p(xt+1|xt)
        Tlogprobs = torch.zeros(T-1, N, N)

        for t in range(T - 1):
            with torch.no_grad():
                xtp1_pred = self._system(x[t], t)
            xtp1_diff = xtp1_pred.unsqueeze(1) - x[None, t + 1]

            Tlogprobs[t] = self._Qpdf.log_prob(xtp1_diff.reshape(-1, n)).reshape(N, N)
	
	# for numerical stability subtract the max
        Tlogprobs -= Tlogprobs.max(1)[0].unsqueeze(1)
        Tprobs = Tlogprobs.exp()

        # compute v
        v = (w[:-1] * Tprobs).sum(1)
        # since Tprobs sum is in the denominator, subtracting Tlogprobs.max
        # above gives the same result as not

        # compute w_N by backward recursion
        w_N = w.clone()
        #  sets w_N[-1] = w[-1]

        for t in range(T - 1):
            t = T - t - 2
            w_N_t = w[t] * (w_N[t + 1] * Tprobs[t] / v[t].unsqueeze(1)).sum(1).unsqueeze(1)
            if w_N_t.sum() > 0.:
                # if no particles have weight just use the filtered weight
                w_N[t] = w_N_t
            # normalize weights
            w_N[t] /= w_N[t].sum()

        return w_N

    def run(self, y):

        x, w = self.filter(y)

        w_N = self.smoother(x, w)

        self._xfilt = x
        self._wfilt = w
        self._wsmooth = w_N

    def get_smooth_mean(self):

        x_mean = (self._xfilt * self._wsmooth).sum(1) / self._wsmooth.sum(1)

        return x_mean


 


    


