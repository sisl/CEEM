from collections import OrderedDict

import torch
from torch import nn

from ceem.dynamics import *


class MagLev(C2DSystem, nn.Module, AnalyticObsJacMixin, DynJacMixin):
    """Simple pendulum, where y = [cos th, sin th]
    """

    def __init__(self, mg, k, dt, method='midpoint'):
        """
           Args:
              mu (torch.tensor): (1,) scalar
        """

        C2DSystem.__init__(self, dt=dt, method=method)
        nn.Module.__init__(self)

        self._mg = nn.Parameter(mg)
        self._k = nn.Parameter(k)

        self._xdim = 2
        self._ydim = 2
        self._udim = 1

    def step_derivs(self, t, x, u):
        q = x[:,:,:1]
        qdot = x[:,:,1:]

        qddot = self._mg - self._k * (u + self._k/self._mg) / q**2

        xdot = torch.cat([qdot,qddot], dim=2)

        return xdot

    def observe(self, t, x, u=None):
        return x

    def jac_obs_x(self, t, x, u=None):
        B,T,n = x.shape
        return torch.eye(n).expand(B,T,n,n)
