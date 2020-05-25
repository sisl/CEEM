from collections import OrderedDict

import torch
from torch import nn

from ceem.dynamics import (AnalyticDynJacMixin, AnalyticObsJacMixin, DiscreteDynamicalSystem,
                           DynJacMixin, ObsJacMixin)
from ceem.nn import LNMLP

DEFAULT_NN = dict(hidden_sizes=[32] * 2, activation='tanh', gain=0.5, ln=False)


class DiscreteLinear(DiscreteDynamicalSystem, nn.Module, AnalyticDynJacMixin, ObsJacMixin):
    """Discrete dynamical system with linear dynamics and linear or non-linear observation model.
    """

    def __init__(self, xdim, udim, ydim, A=None, B=None, C=None, D=None, obsModel=DEFAULT_NN):
        """
            Args:
                xdim (int): dimension of state vector x
                udim (int): dimension of control vector u
                ydim (int): dimension of observation vector y
                A (torch.tensor): (xdim,xdim) model matrix
                B (torch.tensor): (xdim,udim) model matrix
                C (torch.tensor): (ydim,xdim) model matrix
                D (torch.tensor): (ydim,udim) model matrix
                obsModel (dict): dictionary of options for neural net. See nn.py/LNMLP.
                                If None, the observation model only uses C and D.
            """

        super().__init__()
        self._xdim = xdim
        self._ydim = ydim
        self._udim = udim

        # Initialize model matrices
        self._dyn = DynamicsModule(A, B, xdim, udim)
        self._obs = ObservationModule(C, D, obsModel, xdim, udim, ydim)

    def step(self, t, x, u):
        return self._dyn(t, x, u)

    def observe(self, t, x, u):
        y = self._obs(t, x, u)
        return y

    def jac_step_x(self, t, x, u):
        B, T, n = x.shape
        return self._dyn._A.expand(B, T, n, n)

    def jac_step_theta(self, t, x, u=None):
        jacobians = OrderedDict([(name, None) for name, _ in self.named_parameters()])
        B, T, n = x.shape
        jacobians['._dyn._A'] = x.unsqueeze(-1).expand(B, T, self._xdim,
                                                       self._xdim).diag_embed(dim1=-2, dim2=-3)
        jacobians['._dyn._B'] = u.unsqueeze(-1).expand(B, T, self._udim,
                                                       self._udim).diag_embed(dim1=-2, dim2=-3)
        return jacobians


class DynamicsModule(nn.Module):

    def __init__(self, A, B, xdim, udim):
        super().__init__()
        self._A = torch.nn.Parameter(-torch.eye(xdim, xdim) if A is None else A)
        self._B = torch.nn.Parameter(torch.randn(xdim, udim) if B is None else B)

    def forward(self, t, x, u):
        return x @ self._A.t() + u @ self._B.t()


class ObservationModule(nn.Module):

    def __init__(self, C, D, obsModel, xdim, udim, ydim):
        super().__init__()
        self._C = torch.nn.Parameter(torch.randn(ydim, xdim) if C is None else C)
        self._D = torch.nn.Parameter(torch.randn(ydim, udim) if D is None else D)

        # Create observation neural net
        self._net = (lambda x: 0.) if obsModel is None else \
                    LNMLP(input_size=udim + xdim, output_size=ydim, **obsModel)

    def forward(self, t, x, u):
        return x @ self._C.t() + u @ self._D.t() + self._net(torch.cat([x, u], dim=-1))
