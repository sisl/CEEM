import torch
from ceem import utils
from torch.distributions.categorical import Categorical

from torch.distributions.multivariate_normal import MultivariateNormal


from torch.nn.utils import parameters_to_vector, vector_to_parameters

from scipy.optimize import minimize

from ceem import utils


import timeit


class Resampler:

    def __init__(self, method):
        '''
        Class containing Multinomial, Systematic, Stratified resamplers
        Args:
            method (str): 'multinomial', 'stratified', or 'systematic'
        '''
        if method not in ['multinomial', 'stratified', 'systematic']:
            raise NotImplementedError
        else:
            self._method = method

    def __call__(self, weights):
        '''
        Resampling method
        Args:
            weights (torch.tensor): (B, N) particle weights
        Returns:
            indices (torch.tensor): (B, N) particle indices
        '''

        return getattr(self, self._method)(weights)

    def dist(self, weights):
        return Categorical(weights)

    def icdf(self, weights, cds):
        '''
        Linear-time inverse CDF for a categorical distribution
        Args:
            weights (torch.tensor): (B, N) particle weights
            cds (torch.tensor): (B,N) ascending ordered cumulative densities
        Returns:
            indices (torch.tensor): (B, N) particle indices
        '''

        B,N = weights.shape
        indices = torch.zeros(B,N, dtype=torch.long)
        for b in range(B):
            i = 0
            cumden = weights[b,i]
            prev_cds = -1.0
            for n in range(N):
                assert prev_cds < cds[b,n], 'cds must be ascending ordered.'
                prev_cds = cds[b,n]
                while cds[b,n] > cumden:
                    i += 1
                    cumden += weights[b,i]
                indices[b,n] = i

        return indices

    def multinomial(self, weights):

        N = weights.shape[1]
        dist = self.dist(weights)

        return dist.sample((N,)).T

    def stratified(self, weights):

        B,N = weights.shape

        cds = (torch.rand(B,N) + torch.arange(N, dtype=torch.get_default_dtype()).unsqueeze(0))*(1./N)

        print('Stratified')
        print(cds)

        return self.icdf(weights, cds)

    def systematic(self, weights):

        B,N = weights.shape

        cds = (torch.rand(B,1) + torch.arange(N, dtype=torch.get_default_dtype()).unsqueeze(0))*(1./N)

        return self.icdf(weights, cds)


class faPF:

    def __init__(self, N, system, Q, R, Px0, 
        x0mean=None, resampler=Resampler('systematic'), 
        FFBSi_N = None,
        burnin_T = None,
        ess_frac = 0.5,
        transform = lambda x: x):
        """
        Fully-Adapted Particle Filter
        Args:
          N (int):
          system (DiscreteDynamics):
          Q (torch.tensor): (xdim,xdim) torch tensor
          R (torch.tensor): (ydim,ydim) torch tensor
          Px0 (torch.tensor): (xdim,xdim) torch tensor
          x0mean (torch.tensor): (1,xdim) torch tensor
          resampler (Resampler): resampler
          FFBSi_N (int): number of backward trajectories to sample
          burnin_T (int or None): time skipped when computing MCEM Q
          ess_frac (float): Effective Sample Size fraction
          transform (function): function for projecting states onto a manifold
        """

        self._N = N
        self._system = system
        self._xdim = system._xdim
        self._ydim = system._ydim
        self._Q = Q.unsqueeze(0)
        self._Qinv = Q.inverse().unsqueeze(0)
        self._Qdist = MultivariateNormal(torch.zeros(self._xdim), Q)
        self._x0mean = x0mean if x0mean is not None else torch.zeros(1, self._xdim)
        self._Px0 = Px0.unsqueeze(0)
        self._Px0dist = MultivariateNormal(self._x0mean, Px0)
        self._R = R.unsqueeze(0)
        self._Rinv = R.inverse().unsqueeze(0)
        self._Rdist = MultivariateNormal(torch.zeros(self._ydim), R)
        self._resampler = resampler
        self._FFBSi_N = FFBSi_N if FFBSi_N else max(N//10,1)
        self._burnin_T = burnin_T
        self._ess_frac = ess_frac
        self._transform = transform

    def filter(self, y):
        '''
        Run filter
        Args:
            y (torch.tensor): (B,T,m) observartions
        Returns:
            x (torch.tenosr): (B,N,T,n) filtered states
            w (torch.tensor): (B,N,T) filtered weights
        '''

        N = self._N
        B,T,m = y.shape
        x = torch.zeros(B, N, T,  self._xdim)
        xr = torch.zeros_like(x)
        w = torch.ones(B, N, T, dtype=torch.get_default_dtype())/N # (1/N for faPF)  

        # sample initial distribution from p(x0 | y0)
        x[:,:,0:1] = self.sample_initial(y[:,0:1], N)
        x[:,:,0:1] = self._transform(x[:,:,0:1])
        v_unnorm = torch.zeros(B,N,T-1)

        log_prev_v = torch.ones(B,N, dtype=torch.get_default_dtype())

        for t in range(1,T):
            # sample x_t from p(x_t | x_t-1, y_t)
            xtm1 = x[:,:,t-1:t]
            yt = y[:,t:t+1]

            # resample
            resampdist = self.get_resampling_distribution(t-1, xtm1)
            log_vt = resampdist.log_prob(yt.repeat(1,N,1).view(B*N,m)).view(B,N)
           
            v_unnorm[:,:,t-1] = log_vt.exp()
            
            log_vt += log_prev_v
            log_vt -= log_vt.max(dim=1)[0].unsqueeze(1)
            vt = log_vt.exp()
            vt /= vt.sum(dim=1).unsqueeze(1)
            inds = self._resampler(vt.clone())

            xtm1r = xtm1.clone()
            for b in range(B):
                # adaptive
                ESS = 1./(vt[b]**2).sum()
                if ESS < self._ess_frac * N:
                    xtm1r[b] = xtm1[b, inds[b]]
                    log_prev_v[b] = torch.ones(N, dtype=torch.get_default_dtype())
                    # print('Resample at t=%d, ESS=%.3f'%(t,ESS))
                else:
                    log_prev_v[b] = log_vt[b]

            xr[:,:,t-1:t] = xtm1r

            # propegate
            sampdist = self.get_sampling_distribution(t-1, xtm1r, yt)

            xt = sampdist.sample().reshape(B,N,1,self._xdim)
            xt = self._transform(xt)

            x[:,:,t:t+1] = xt

        mean_ll = v_unnorm.mean(1).log().mean()

        return x, xr, w, mean_ll


    def sample_initial(self, y0, N):
        '''
        Sample from p(x0 | y0) 
        Args:
            y0 (torch.tensor): (B,1,m) initial observation
        Returns:
            x0 (torch.tensor): (B,N,1,n) samples of x0
        Notes: 
            Eqns 352-354 https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf
        '''
        B, _, m = y0.shape
        y0 = y0.squeeze(1)

        mu_x = self._x0mean.repeat(B,1)

        tinp = torch.tensor([0] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
        C = self._system.jac_obs_x(tinp, mu_x.unsqueeze(1)).detach().squeeze(1)

        Sig_c = (C @ self._Px0).transpose(-2,-1)

        Sig_y = self._R + C @ self._Px0 @ C.transpose(-2,-1)

        muhat_x = mu_x + (Sig_c @ torch.solve(
            y0.unsqueeze(-1) - C @ mu_x.unsqueeze(-1), Sig_y)[0]).squeeze(-1)
        Sighat_x = self._Px0 - Sig_c @ torch.solve(Sig_c.transpose(-2,-1), Sig_y)[0]

        x0dist = MultivariateNormal(muhat_x, Sighat_x)

        # xtest = self._Px0dist.sample((100,))

        # ll_y_x = self._Rdist.log_prob((C @ xtest.transpose(-2,-1)).squeeze(-1) - y0).unsqueeze(-1)
        # ll_x = self._Px0dist.log_prob(xtest)
        # joint_ll = ll_y_x + ll_x

        # ll_x_y = x0dist.log_prob(xtest)

        # diff = ll_x_y - joint_ll

        # above diff is constant - implying this math is correct


        return x0dist.sample((N,)).transpose(0,1).unsqueeze(2)



    def get_sampling_distribution(self, t, xtm1, yt):
        '''
        Sampling distribution p(xt | xt-1, yt)
        Args:
            t (int): time-index
            xtm1 (torch.tensor): (B,N,1,n) state-particles
            yt (torch.tensor): (B,1,m) observation 
        Returns:
            dist (MultivariateGaussian): p(xt | xt-1, yt)
        Notes:
            Implementation of
            https://link.springer.com/content/pdf/10.1023%2FA%3A1008935410038.pdf
            Equation 18-20
        '''

        B,N,_,xdim = xtm1.shape
        m = yt.shape[-1]
        tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())

        fxtm1 = self._system.step(tinp-1, xtm1.view(B*N,1,xdim))
        Ht = self._system.jac_obs_x(t, fxtm1).detach().squeeze(1)
        HtT = Ht.transpose(-2,-1)

        yt_exp = self._system.observe(tinp, fxtm1)
        fxtm1 = fxtm1.squeeze(1)

        Sigtinv = self._Qinv + HtT @ self._Rinv @ Ht
        Sig = Sigtinv.inverse()

        yt_ = yt.repeat(1,N,1).view(B*N,m,1) 
        yt_ += Ht @ fxtm1.unsqueeze(-1) - yt_exp.transpose(-2,-1)

        mt = Sig @ (self._Qinv @ fxtm1.unsqueeze(-1) + HtT @ self._Rinv @ yt_)
        mt = mt.squeeze(-1)

        return MultivariateNormal(mt, Sig)

    def get_resampling_distribution(self, t, xtm1):
        '''
        Resampling distribution p(yt | xt-1)
        Args:
            t (int): time-index
            xtm1 (torch.tensor): (B,N,1,n) state-particles
        Returns:
            dist (MultivariateGaussian): p(yt | xt-1)
        Notes:
            Implementation of
            https://link.springer.com/content/pdf/10.1023%2FA%3A1008935410038.pdf
            Equation 14
        '''
        ydim = self._system.ydim

        B,N,_,xdim = xtm1.shape

        tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())

        fxtm1 = self._system.step(tinp-1, xtm1.view(B*N,1,xdim))
        Ht = self._system.jac_obs_x(t, fxtm1).detach().squeeze(1)
        HtT = Ht.transpose(-2,-1)

        yt = self._system.observe(tinp, fxtm1)

        mt = yt.squeeze(1)
        Sig = (self._R + Ht @ self._Q @ HtT)

        return MultivariateNormal(mt, Sig)

    '''
    Smoothing Methods
    '''
    def FFBSm(self, y, x, w, return_wij_tN=False):
        '''
        Compute the smoothing marginal distributions
        Args:
            y (torch.tensor): (B,T,m) observartions
            x (torch.tenosr): (B,N,T,n) filtered states
            w (torch.tensor): (B,N,T) filtered weights
            return_wij_tN (bool): return pair-wise smoothing weights
        Returns:
            xsm (torch.tenosr): (B,N,T,n) smoothed states
            wsm (torch.tensor): (B,N,T) smoothed weights
            wij_tN (torch.tensor): (B,N,N,T) pairwise smoothing weights
        Notes:
            Implemenation of Forward Filtering-Backward Smoothing
            https://www.stats.ox.ac.uk/~doucet/doucet_johansen_tutorialPF2011.pdf
            Equation 49
        '''

        B,N,T,n = x.shape

        wsm = w.clone()
        wsm /= wsm.sum(1).unsqueeze(1) # normalize
        wij_tN = torch.zeros(B,N,N,T-1)

        for t in range(T-1):
            tau = T - t - 2

            # compute p(X_tau+1 | X_tau)
            tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
            fxtau = self._system.step(tinp, x[:,:,tau])
            dxtaup1 = x[:,:,tau+1].unsqueeze(2) - fxtau.unsqueeze(1) # (B,N,N,n)
            fprobs = self._Qdist.log_prob(dxtaup1.view(B*N*N,n)).view(B,N,N).exp()

            den = (fprobs * w[:,:,tau:tau+1]).sum(-2).unsqueeze(-2)
            
            wij_tN[:,:,:,tau] = wsm[:,:,tau].unsqueeze(-1) * (wsm[:,:,tau+1:tau+2].transpose(-2,-1) * fprobs / den)
            
            wsm[:,:,tau] = wij_tN[:,:,:,tau].sum(-1)
            wsm[:,:,tau] /= wsm[:,:,tau].sum(1).unsqueeze(1) # normalize

            wij_tN[:,:,:,tau] /= wij_tN[:,:,:,tau].sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)

        xsm = x.clone()

        if return_wij_tN:
            return xsm, wsm, wij_tN
        else:
            return xsm, wsm

    def FFBSi(self, x, w):
        '''
        Sample trajectories from JSD using FFBSi
        Args:
            x (torch.tenosr): (B,N,T,n) filtered states
            w (torch.tensor): (B,N,T) filtered weights
        Returns:
            xsamps (torch.tesnor): (B, Ns, T, n) sampled state trajs
        Notes: 
            See http://users.isy.liu.se/rt/schon/Publications/LindstenS2013.pdf
            Algorithm 4
        '''

        B,N,T,n = x.shape
        Ns = self._FFBSi_N

        xsamps = torch.zeros((B, Ns, T, n))

        # Sample at T
        bj = Categorical(w[:,:,-1]).sample((Ns,)).T

        for b in range(B):
            xsamps[b,:,-1] = x[b, bj[b],-1]


        for tau in range(T-1):
            t = T - tau - 2

            tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
            fxt = self._system.step(tinp, x[:,:,t])

            dxtp1 = xsamps[:,:,t+1].unsqueeze(2) - fxt.unsqueeze(1) # (B,Ns,N,n)
            flgprobs = self._Qdist.log_prob(dxtp1.view(B*Ns*N,n)).view(B,Ns,N)

            wlgfprobs = flgprobs + w[:,:,t].unsqueeze(1).log()

            bj = Categorical(logits=flgprobs).sample()

            for b in range(B):
                xsamps[b,:,t] = x[b, bj[b],t]

        return xsamps

    '''
    Methods for computing Q(theta | theta_k)
    '''
    def Q_MCEM(self, y, x):
        '''
        Compute Monte-Carlo appx Q(theta, theta_k) 
                using iid x_1:T ~ p(x_1:T|y_1:T)
        Args:
            y (torch.tensor): (B,T,m) observartions
            x (torch.tenosr): (B,N,T,n) sampled state trajs
        Returns:
            Q (torch.tensor): (1,) Q(theta, theta_k)
        '''

        B,N,T,n = x.shape
        m = self._system.ydim

        burnin_T = self._burnin_T if self._burnin_T is not None else T//10
        Ttil = T - burnin_T

        x = x.detach()
        y = y.detach()

        # Compute I1
        if burnin_T == 0:
            I1 = self._Px0dist.log_prob(x[:,:,0].view(B*N,n)).view(B,N)
            I1 = I1.sum()
        else:
            I1 = 0.

        # Compute approx I2
        tinp = torch.stack([torch.arange(burnin_T, T-1,dtype=torch.get_default_dtype())]*B, dim=0)
        fx = self._system.step(tinp, x[:,:,burnin_T:-1].reshape(B, N*(Ttil-1),n)).view(B*N*(Ttil-1),n)
        dx = x[:,:,burnin_T+1:].reshape(B*N*(Ttil-1),n) - fx
        I2 = self._Qdist.log_prob(dx).sum()


        # Compute I3
        tinp = torch.stack([torch.arange(burnin_T, T,dtype=torch.get_default_dtype())]*B, dim=0)
        tinp = tinp.unsqueeze(1).repeat((1,N,1)).view(B,Ttil*N)
        gx = self._system.observe(tinp, x[:,:,burnin_T:].reshape(B,N*Ttil,n)).view(B,N,Ttil,m)
        dy = y[:,burnin_T:].unsqueeze(1) - gx
        glogprobs = self._Rdist.log_prob(dy.reshape(B*N*Ttil,m)).view(B,N,Ttil)
        I3 = glogprobs.sum()

        return (I1 + I2 + I3) / N

    def Q_JSD_marginals(self, y, x, w, wij_tN):
        '''
        Compute Q(theta, theta_k) using approximate JSD marginals
        Args:
            y (torch.tensor): (B,T,m) observartions
            x (torch.tenosr): (B,N,T,n) filtered states
            w (torch.tensor): (B,N,T) filtered weights
            wij_tN (torch.tensor): (B,N,N,T-1) pair-wise smoothing weights
        Returns:
            Q (torch.tensor): (1,) Q(theta, theta_k)
        Notes:
            See http://user.it.uu.se/~thosc112/pubpdf/schonwn2011-2.pdf
            Equations 48-49
        '''

        if torch.any(torch.isnan(wij_tN)):
            import ipdb
            ipdb.set_trace()

        B,N,T,n = x.shape
        m = self._system.ydim

        x = x.detach()
        y = y.detach()
        w = w.detach()
        wij_tN = wij_tN.detach()

        # Compute I1
        I1 = self._Px0dist.log_prob(x[:,:,0].view(B*N,n)).view(B,N)
        I1 = (I1 * w[:,:,0]).sum()

        # Compute I2
        tinp = torch.stack([torch.arange(T-1,dtype=torch.get_default_dtype())]*B, dim=0)
        tinp = tinp.unsqueeze(1).repeat((1,N,1)).view(B,(T-1)*N)
        fx = self._system.step(tinp, x[:,:,:-1].reshape(B,N*(T-1),n)).view(B,N,T-1,n)
        dfx = x[:,:,:-1].unsqueeze(2) - fx.unsqueeze(1)
        flogprobs = self._Qdist.log_prob(dfx.view(B*N*N*(T-1),n)).view(B,N,N,T-1)
        I2 = (flogprobs * wij_tN).sum()

        # Compute I3
        tinp = torch.stack([torch.arange(T,dtype=torch.get_default_dtype())]*B, dim=0)
        tinp = tinp.unsqueeze(1).repeat((1,N,1)).view(B,T*N)
        gx = self._system.observe(tinp, x.reshape(B,N*T,n))
        dy = y.unsqueeze(1) - gx
        glogprobs = self._Rdist.log_prob(dy.view(B*N*T,m)).view(B,N,T)
        I3 = (glogprobs * w).sum()

        return I1 + I2 + I3


    '''
    Misc methods
    '''

    def fa_loglikelihood(self, y, xfilt):
        '''
        Estimates log-likehood of filtered particles from faPF
        Eqn 39 from https://arxiv.org/pdf/1703.02419.pdf
        Args:
            y (torch.tensor): (B,T,m) observartions
            xfilt (torch.tenosr): (B,N,T,n) filtered states
        Returns:
            z (torch.tensor): (1,) log p(y_1:T | x_filt, theta)
        '''

        B,N,T,n = xfilt.shape
        m = self._system._ydim

        xfilt = xfilt.detach()
        y = y.detach()

        # propegate
        tinp = torch.stack([torch.arange(T-1,dtype=torch.get_default_dtype())]*B, dim=0)
        fx = self._system.step(tinp, xfilt[:,:,:-1].reshape(B, N*(T-1),n))
        yfx = self._system.observe(tinp, fx).view(B,N,T-1,m)
        dy = y[:,1:].unsqueeze(1) - yfx
        v = self._Rdist.log_prob(dy.view(-1,m)).view(B,N,T-1).exp()
        vmean = v.mean(1)
        log_z = vmean.log().sum(-1).mean()

        return log_z
        


    def compute_mean(self, x, w):
        '''
        Computes mean trajectory from weighted samples
        Args:
            x (torch.tenosr): (B,N,T,n) filtered states
            w (torch.tensor): (B,N,T) filtered weights
        Returns
            xmean (torch.tensor): (B,T,n) mean states
        '''

        # ensure normalized
        w = w.clone() / w.sum(1).unsqueeze(1)

        w = w.unsqueeze(-1)

        return (x * w).sum(1)

from ceem import logger

def HarmonicDecayScheduler(k, a=50.0):
    a = float(a)
    return a/(k+a) 


def torchoptimizer(objfun, params, lr=1e-2, nepochs=10):

    opt = torch.optim.Adam(params, lr=lr)

    for epoch in range(nepochs):
        opt.zero_grad()

        loss = objfun()

        loss.backward()

        opt.step()

    return


def scipyoptimizer(objfun, params, method='BFGS'):

    vparams0 = parameters_to_vector(params).clone()

    def eval_f(vparams):
        vparams = torch.tensor(vparams)

        vparams_ = parameters_to_vector(params)
        vector_to_parameters(vparams, params)

        with torch.no_grad():
            obj = objfun()

        vector_to_parameters(vparams_, params)

        return obj.detach().numpy()

    def eval_g(vparams):
        vparams = torch.tensor(vparams)

        vparams_ = parameters_to_vector(params)
        vector_to_parameters(vparams, params)

        obj = objfun()

        obj.backward()

        grads = torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in params
        ])

        grads = grads.detach().numpy()

        vector_to_parameters(vparams0, params)

        return grads


    with utils.Timer() as time:
        if method == 'BFGS':
            opt_result_ = minimize(eval_f, vparams0.detach().numpy(), 
                jac=eval_g, method='BFGS', options={'disp':True})
        elif method == 'Nelder-Mead':
            opt_result_ = minimize(eval_f, vparams0.detach().numpy(), 
                method='Nelder-Mead', options={'disp':True})

    vparams = opt_result_['x']

    return torch.tensor(vparams)

class SAEMTrainer:

    def __init__(self, fapf, y, 
            gamma_sched=HarmonicDecayScheduler,
            optimizer=torchoptimizer,
            max_k=100,
            xlen_cutoff=None):
        '''
        Stochastic-Approximation EM trainer
        Args:
            fapf (faPF): Fully-Adapted Particle Filter
            y (torch.tensor): (B,T,m) observation time-series
        Notes: 
            https://projecteuclid.org/download/pdf_1/euclid.aos/1018031103 
        '''
        self._fapf = fapf
        self._optimizer=optimizer
        self._y = y
        self._gamma_sched = gamma_sched
        self._max_k = max_k
        self._xlen_cutoff = xlen_cutoff

    def train(self, params, callbacks=[]):

        vparams0 = parameters_to_vector(params).clone()

        xsms = []

        t_start = timeit.default_timer()

        for k in range(self._max_k):

            with utils.Timer() as time:
                ## E-step
                xfilt, xfiltr, wfilt, meanll = self._fapf.filter(self._y)
                xsm = self._fapf.FFBSi(xfilt, wfilt)
                xsms.append(xsm)
                if self._xlen_cutoff:
                    if len(xsms) > self._xlen_cutoff:
                        xsms = xsms[-self._xlen_cutoff:]

            logger.logkv('train/Etime', time.dt)

            ## M-step

            with utils.Timer() as time:
                obj = lambda: -self.recursive_Q(xsms, self._y, 0, 0.)

                self._optimizer(obj, params)

            logger.logkv('train/Mtime', time.dt)

            logger.logkv('train/elapsedtime', timeit.default_timer() - t_start)


            ## log the current value of Q

            Q = float(self._fapf.Q_MCEM(self._y, xsms[-1]))
            logger.logkv('train/Q', Q)


            for callback in callbacks:
                callback(k)

            logger.dumpkvs()

        return params

    def recursive_Q(self, xs, y, k_, Q):
        '''
        Recursive computation of Q
        Args:
            x (list of torch.tensor): [(B,N,T,n)] iid smoothed trajs
            y (torch.tensor): (B,T,m) observations
            k_ (int): call level
            Q (torch.tensor): (1,) Q(theta) or 0.
        Returns:
            Q (torch.tensor): (1,) Q(theta)
        '''

        if len(xs) == 0:
            return Q
        else:
            gam = self._gamma_sched(k_)
            Q_ = self._fapf.Q_MCEM(y, xs[0])
            Qk = Q * (1-gam) + gam * Q_
            return self.recursive_Q(xs[1:], y, k_+1, Qk) 



