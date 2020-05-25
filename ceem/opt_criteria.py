#
# File: opt_criteria.py
#
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.utils import parameters_to_vector

from ceem import logger
from ceem.dynamics import (AnalyticDynJacMixin, AnalyticObsJacMixin, DynJacMixin, ObsJacMixin)



class Criterion:

    def __call__(self, model, x, **kwargs):
        return self.forward(model, x, **kwargs)

    def batched_forward(self, model, x, **kwargs):
        """
        Forward method for computing criterion, not summed over batch
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
        Returns:
            y (torch.tensor): (B,) criterion
        """
        B,T,n = x.shape
        y = torch.stack([self(model,
            x[i:i+1],**kwargs) for i in range(B)])
        return y

    def batched_sample_forward(self, model, x, **kwargs):
        """
        Forward method for computing criterion, not summed over batch
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,N,T,n) system states
        Returns:
            y (torch.tensor): (B,N) criterion
        """
        B,N,T,n = x.shape
        ys = [self.batched_forward(model, x[:,i], **kwargs) for i in range(N)]
        return torch.stack(ys, dim=1)


    def forward(self, model, x, **kwargs):
        """
        Forward method for computing criterion
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
        Returns:
            criterion (torch.tensor): scalar criterion
        """
        raise NotImplementedError

    def jac_x(self, model, x, **kwargs):
        """
        Method for computing jacobian of criterion wrt x
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
        Returns:
            jac_x (torch.tensor): (B,T,n) criterion jacobian
        """

        x.requires_grad = True
        criterion = self.forward(model, x, **kwargs)

        jac_x = grad(criterion, x)[0]

        return jac_x

    def jac_theta(self, model, x, **kwargs):
        """
        Method for accumulating criterion jacobian on parameters
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
            return_grad (bool): if True, returns grad else accumulates to leaf nodes
        """
        criterion = self.forward(model, x, **kwargs)

        return grad(criterion, model.parameters(), allow_unused=True)


class GroupCriterion(Criterion):
    """
    A group of Criterion
    """

    def __init__(self, criteria):
        self._criteria = criteria

    def forward(self, model, x, **kwargs):
        return sum([c(model, x, **kwargs) for c in self._criteria])


class STRParamCriterion(Criterion):
    """
    Soft Trust Region on params criterion
    """

    def __init__(self, rho, params):
        self._rho = rho
        self._params = params
        self._vparams0 = parameters_to_vector(params).clone().detach()

    def forward(self, model, x, **kwargs):
        vparams = parameters_to_vector(self._params)
        return self._rho * torch.sum((vparams - self._vparams0)**2)


class SOSCriterion(Criterion):
    """
    Sum-of-squares criterion
    """

    def forward(self, model, x, **kwargs):
        return 0.5 * (self.residuals(model, x, **kwargs)**2).sum()

    def residuals(self, model, x, **kwargs):
        """
        Forward method for computing SOS residuals
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
        Returns:
            residuals (torch.tensor): (nresid,) residuals
        """
        raise NotImplementedError

    def scaled_jac_x_diag(self, model, x, sparse=False):
        """
        Returns the diagonal blocs of the Jacobian
        Args:
            x (torch.tensor): (B,T,n) system states
        Returns
            jac_resid_x (lambda): lambda function of (b,t) that returns 2 torch.tensors of dimension (n,n)
        """

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix, **kwargs):
        """
        Method for computing residuals jacobian wrt x
        Args:
            model (DiscreteDynamicalSystem)
            x (torch.tensor): (B,T,n) system states
            sparse (bool): if True returns a (nresid,B*T*n) sparse_format
            sparse_format (scipy.sparse): sparse matrix format
        Returns:
            jac_resid_x (torch.tensor): (nresid,B,T,n) criterion jacobian
        """
        logger.warn('Using a very inefficient jac_resid_x')

        x = x.detach()
        x.requires_grad_(True)

        resid = self.residuals(model, x, **kwargs)

        jac_resid_x = []

        for r in resid:
            jac_resid_x.append(grad(r, x, retain_graph=True)[0])

        jac_resid_x = torch.stack(jac_resid_x, dim=0)

        nresid, B, T, n = jac_resid_x.shape

        if sparse:
            return sp.csc_matrix(
                jac_resid_x.view(nresid, B * T * n).detach().numpy(), dtype=np.float64)
        else:
            return jac_resid_x


class GaussianObservationCriterion(SOSCriterion):

    def __init__(self, Sig_y_inv, t, y, u=None):

        self._t = t

        self._y = y

        self._u = u

        self._size = Sig_y_inv.shape[0]

        if Sig_y_inv.ndim == 1:
            # assuming diagonal
            self._Sig_y_inv_chol = Sig_y_inv.sqrt().unsqueeze(0).unsqueeze(0)
            self._diagcov = True
        elif Sig_y_inv.ndim == 2:
            # assuming full
            self._Sig_y_inv_chol = Sig_y_inv.cholesky().unsqueeze(0).unsqueeze(0)
            self._diagcov = False
        elif Sig_y_inv.ndim == 3:
            # assuming full, timevarying
            self._Sig_y_inv_chol = Sig_y_inv.cholesky().unsqueeze(0)
            self._diagcov = False

    def apply_inds(self, x, inds):
        if inds is not None:
            t = self._t[inds]
            x = x[inds]
            y = self._y[inds]
            u = self._u[inds] if self._u is not None else None
        else:
            t = self._t
            y = self._y
            u = self._u
        return t, x, y, u

    def residuals(self, model, x, inds=None, flatten=True):

        t, x, y, u = self.apply_inds(x, inds)

        ypred = model.observe(t, x, u)

        err = ypred - y

        if self._diagcov:
            resid = self._Sig_y_inv_chol * err
        else:
            resid = (self._Sig_y_inv_chol @ err.unsqueeze(-1)).squeeze(-1)

        return resid.view(-1) if flatten else resid

    def scaled_jac_x_diag(self, model, x, inds=None):
        t, x, y, u = self.apply_inds(x, inds)
        jac_obs_x = model.jac_obs_x(t, x, u)

        if self._diagcov:
            jac_resid_x_ = self._Sig_y_inv_chol.unsqueeze(-1) * jac_obs_x
        else:
            jac_resid_x_ = self._Sig_y_inv_chol @ jac_obs_x

        J = jac_resid_x_.detach()
        return J

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix, inds=None):
        t, x, y, u = self.apply_inds(x, inds)
        if isinstance(model, ObsJacMixin):
            jac_resid_x_ = self.scaled_jac_x_diag(model, x)
            B, T, m, n = jac_resid_x_.shape
            # currently (B,T,m,n)

            if sparse:
                idx = torch.arange(B * T)
                idptr = torch.arange(B * T + 1)
                jac_resid_x = sp.bsr_matrix((jac_resid_x_.view(B * T, m, n), idx, idptr),
                                            shape=(T * m * B, T * n * B), dtype=np.float64)
                return sparse_format(jac_resid_x, dtype=np.float64)
            else:
                jac_resid_x = torch.zeros(B * T * m, B, T, n)
                for b in range(B):
                    for t in range(T):
                        jac_resid_x[(b * T + t) * m:(b * T + t + 1) *
                                    m, b, t, :] = jac_resid_x_[b, t]
                return jac_resid_x

        else:
            print('model needs an ObsJacMixin')
            raise NotImplementedError


class STRStateCriterion(SOSCriterion):
    """
    Soft Trust Region on states 
    """

    def __init__(self, rho, x0):

        self._size = x0.shape[-1]
        self._rho = rho
        self._x0 = x0

    def residuals(self, model, x, inds=None, flatten=True):
        res = self._rho * (x - self._x0)
        return res.view(-1) if flatten else res

    def scaled_jac_x_diag(self, model, x, inds=None):
        B, T, n = x.shape
        jac_resid = (self._rho * torch.eye(n).unsqueeze(0).unsqueeze(0)).repeat(B, T, 1, 1)
        return jac_resid.detach()

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix):
        B, T, n = x.shape
        if sparse:
            return self._rho * sp.eye(B * T * n, format=sparse_format.format, dtype=np.float64)
        else:
            return self._rho * torch.eye(B * T * n).to(dtype=x.dtype).view(B * T * n, B, T, n)

class GaussianX0Criterion(SOSCriterion):
    """
    Soft Trust Region on states 
    """

    def __init__(self, x0, Sig_x0_inv):

        self._size = x0.shape[-1]
        self._x0 = x0
        if Sig_x0_inv.ndim == 1:
            # assuming diagonal
            self._Sig_x0_inv_chol = Sig_x0_inv.sqrt().unsqueeze(0).unsqueeze(0)
            self._diagcov = True
        else:
            self._Sig_x0_inv_chol = Sig_x0_inv.cholesky().unsqueeze(0).unsqueeze(0)
            self._diagcov = False

    def residuals(self, model, x, inds=None, flatten=True):
        err = (x[:,0] - self._x0)

        if self._diagcov:
            resid = self._Sig_x0_inv_chol * err
        else:
            resid = (self._Sig_x0_inv_chol @ err.unsqueeze(-1)).squeeze(-1)

        return resid.view(-1) if flatten else resid

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix):
        B, T, n = x.shape

        if self._diagcov:
            cov = self._Sig_x0_inv_chol.squeeze().diag()
        else:
            cov = self._Sig_x0_inv_chol
        # cov = cov.unsqueeze(0).repeat(B,1,1)

        retval = torch.zeros(B*n, B, T, n, dtype=x.dtype)
        for b in range(B):
            retval[b*n:(b+1)*n,b,0,:] = cov
        # retval[:,:,0,:] = cov.view(B*n,B,n)

        if sparse:
            return sparse_format(retval.view(-1,B*T*n).detach().numpy(), dtype=np.float64)
        else:
            return retval


class GaussianDynamicsCriterion(SOSCriterion):

    def __init__(self, Sig_w_inv, t, u=None):

        self._t = t
        self._u = u
        self._size = Sig_w_inv.shape[0]

        if Sig_w_inv.ndim == 1:
            # assuming diagonal
            self._Sig_w_inv_chol = Sig_w_inv.sqrt().unsqueeze(0).unsqueeze(0)
            self._diagcov = True
        else:
            self._Sig_w_inv_chol = Sig_w_inv.cholesky().unsqueeze(0).unsqueeze(0)
            self._diagcov = False

    def apply_inds(self, x, inds):
        if inds is not None:
            t = self._t[inds]
            x = x[inds]
            u = self._u[inds] if self._u is not None else None
        else:
            t = self._t
            u = self._u
        return t, x, u

    def residuals(self, model, x, inds=None, flatten=True):
        t, x, u = self.apply_inds(x, inds)

        u = u[:, :-1] if u is not None else None
        xpred = model.step(self._t[:, :-1], x[:, :-1], u)

        err = xpred - x[:, 1:]

        if self._diagcov:
            resid = self._Sig_w_inv_chol * err
        else:
            resid = (self._Sig_w_inv_chol @ err.unsqueeze(-1)).squeeze(-1)

        return resid.view(-1) if flatten else resid

    def scaled_jac_x_diag(self, model, x, inds=None):
        t, x, u = self.apply_inds(x, inds)

        u = u[:, :-1] if u is not None else None
        jac_dyn_x_ = model.jac_step_x(t[:, :-1], x[:, :-1], u)

        if self._diagcov:
            jac_resid_x_ = self._Sig_w_inv_chol.unsqueeze(-1) * jac_dyn_x_
            neyew = -torch.diag_embed(self._Sig_w_inv_chol.view(-1))
        else:
            jac_resid_x_ = self._Sig_w_inv_chol @ jac_dyn_x_
            neyew = -self._Sig_w_inv_chol.view(-1)

        J = jac_resid_x_.detach()

        return J, neyew

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix, inds=None):
        t, x, u = self.apply_inds(x, inds)

        if isinstance(model, DynJacMixin):
            B, T, n = x.shape
            u = u[:, :-1] if u is not None else None
            jac_dyn_x_ = model.jac_step_x(t[:, :-1], x[:, :-1], u)

            jac_resid_x_, neyew = self.scaled_jac_x_diag(model, x)

            # currently (B,T-1,n,n)

            if sparse:
                # Create the right indices given bloc diagonal
                # There is a slight subtlety as intersection between batches
                idx_ = torch.arange((T - 1))
                idx = torch.cat([idx_ + b * T for b in range(B)])
                idptr = torch.arange(B * (T - 1) + 1)

                # Create diagonal matrix with jacobians
                J1 = sp.bsr_matrix((jac_resid_x_.view(B * (T - 1), n, n), idx, idptr),
                                   shape=((T - 1) * n * B, T * n * B), dtype=np.float64)

                # Create diagonal matrix with negative identity
                E = (neyew.unsqueeze(0).repeat(B * (T - 1), 1, 1))
                J2 = sp.bsr_matrix((E, idx + 1, idptr), shape=((T - 1) * n * B, T * n * B),
                                   dtype=np.float64)

                return sparse_format(J1 + J2, dtype=np.float64)
            else:
                jac_resid_x = torch.zeros(B * (T - 1) * n, B, T, n)
                for b in range(B):
                    for t in range(T - 1):
                        jac_resid_x[(b * (T - 1) + t) * n:(b * (T - 1) + t + 1) *
                                    n, b, t, :] = jac_resid_x_[b, t]
                        jac_resid_x[(b * (T - 1) + t) * n:(b * (T - 1) + t + 1) * n, b, t +
                                    1, :] = neyew
                return jac_resid_x

        else:
            return super().jac_resid_x(model, t, x)


class BasicGroupSOSCriterion(SOSCriterion):
    """
    Group of SOSCriterion instances
    """

class GroupSOSCriterion(BasicGroupSOSCriterion):
    """
    Group of SOSCriterion instances
    """

    def __init__(self, criteria):

        self._criteria = criteria

    def residuals(self, model, x, **kwargs):
        residuals_list = [c.residuals(model, x, **kwargs) for c in self._criteria]

        return torch.cat(residuals_list, dim=0)

    def jac_resid_x(self, model, x, sparse=False, sparse_format=sp.csr_matrix, **kwargs):

        jacs = [
            c.jac_resid_x(model, x, sparse=sparse, sparse_format=sp.csr_matrix, **kwargs)
            for c in self._criteria
        ]

        B, T, n = x.shape

        if sparse:
            if sparse_format is not sp.coo_matrix:
                return sparse_format(sp.vstack(jacs), dtype=np.float64)
            else:
                return sp.vstack(jacs)
        else:
            return torch.cat(jacs, dim=0)


class BlockSparseGroupSOSCriterion(BasicGroupSOSCriterion):
    """
    Group of SOSCriterion instances with careful design of the sparsity pattern in the Jacobian matrix.
    All SOSCriterion but the last one must have block diagonal jacobians and a `.scaled_jac_x_diag` method.
    The last SOSCriterion, typically the dynamics, has two blocks on the primary and first upper diagonal.
    """

    def __init__(self, criteria):

        self._criteria = criteria
        self._size = sum([c._size for c in criteria])
        self._mem = {}

    def residuals(self, model, x, **kwargs):
        B, T, n = x.shape
        residuals = torch.zeros(B, T, self._size)

        # Assign all residuals of length T residuals
        s = 0

        for c in self._criteria[:-1]:
            residuals[:, :, s:s + c._size] = c.residuals(model, x, **kwargs, flatten=False)
            s += c._size

        # Assign dynamics residuals (length T-1)
        residuals[:, :-1, -n:] = self._criteria[-1].residuals(model, x, **kwargs, flatten=False)

        return residuals.view(-1)

    def jac_resid_x(self, model, x, sparse_format=sp.bsr_matrix, **kwargs):
        B, T, n = x.shape

        # Check for memoized sparsity pattern
        key = f'{B},{T},{n}'
        if key in self._mem:
            pattern = self._mem[key]
        else:
            pattern = {}
            # For J1
            idx_ = torch.arange(T)
            pattern['idx1'] = torch.cat([idx_ + b * T for b in range(B)])
            pattern['idptr1'] = torch.arange(B * T + 1)

            # For J2
            pattern['idx2'] = pattern['idx1'].view(B, T)[:, 1:].reshape(-1)
            idptr_ = torch.arange(T)
            pattern['idptr2'] = torch.cat([idptr_ + b * (T - 1) for b in range(B)] +
                                          [torch.tensor([B * (T - 1)])])

            self._mem[key] = pattern

        # Obtain building blocks from subcriteria
        J = []
        for i in range(len(self._criteria) - 1):
            J.append(self._criteria[i].scaled_jac_x_diag(model, x))
        m = self._size
        Jd, E = self._criteria[-1].scaled_jac_x_diag(model, x)
        Jd = F.pad(Jd, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)

        # Build main diagonal
        D = torch.cat(J + [Jd], 2)
        J1 = sp.bsr_matrix((D.view(-1, m, n), pattern['idx1'], pattern['idptr1']),
                           shape=(B * T * m, B * T * n), dtype=np.float64)

        # Add off-diag term
        E = (E.unsqueeze(0).unsqueeze(0).repeat(B, (T - 1), 1, 1))

        E = F.pad(E, (0, 0, m - n, 0), mode='constant', value=0).view(B * (T - 1), m, n)

        J2 = sp.bsr_matrix((E, pattern['idx2'], pattern['idptr2']), shape=(B * T * m, B * T * n),
                           dtype=np.float64)

        return sparse_format(J1 + J2)
