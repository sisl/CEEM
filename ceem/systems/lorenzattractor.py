from collections import OrderedDict

import torch
from torch import nn

from ceem.dynamics import *


class LorenzAttractor(C2DSystem, nn.Module, AnalyticObsJacMixin, DynJacMixin):
    """Basic Lorenz Attractor
    """

    def __init__(self, sigma, rho, beta, C, dt, method='midpoint'):
        """
           Args:
              sigma (torch.tensor): (1,) scalar
              rho (torch.tensor): (1,) scalar
              beta (torch.tensor): (1,) scalar
              C (torch.tensor): (ydim, n) observation matrix
        """

        C2DSystem.__init__(self, dt=dt, method=method)
        nn.Module.__init__(self)

        self._sigma = nn.Parameter(sigma)
        self._rho = nn.Parameter(rho)
        self._beta = nn.Parameter(beta)

        # self._C = nn.Parameter(C.unsqueeze(0).unsqueeze(0))

        self._C = C.unsqueeze(0).unsqueeze(0)  # not a learned parameter

        self._xdim = 3
        self._ydim = C.shape[0]
        self._udim = None

    def step_derivs(self, t, x, u=None):
        x = x.to(self._C.dtype)
        x_ = x[:, :, 0:1]
        y_ = x[:, :, 1:2]
        z_ = x[:, :, 2:3]

        xdot = self._sigma * (y_ - x_)
        ydot = x_ * (self._rho - z_) - y_
        zdot = x_ * y_ - self._beta * z_

        inpdot = torch.cat([xdot, ydot, zdot], dim=2)

        return inpdot

    def observe(self, t, x, u=None):
        x = x.to(self._C.dtype)
        return (self._C @ x.unsqueeze(3)).squeeze(3)

    def jac_obs_x(self, t, x, u=None):
        B, T, n = x.shape
        return self._C.repeat(B, T, 1, 1)

    def jac_obs_theta(self, t, x, u):
        # observation doesnt depend on learned params
        jacobians = OrderedDict([
            ('_sigma', None),
            ('_rho', None),
            ('_beta', None),
        ])
        return jacobians


def default_lorenz_attractor(seed=4, obsdif=1, dt=0.04):

    currng = torch.random.get_rng_state()

    torch.manual_seed(seed)

    n = 3

    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3])

    obsdim = n - obsdif
    C = torch.randn(obsdim, n)

    true_system = LorenzAttractor(sigma, rho, beta, C, dt, method='midpoint')

    torch.random.set_rng_state(currng)

    return true_system


def main():
    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    csys = LorenzAttractor(sigma, rho, beta, C)

    x = torch.randn(2, 2, 3)

    print(csys.step(None, x, None).shape)

    print(csys.jac_dyn_x(None, x, None))
    print(csys.jac_dyn_theta(None, x, None))
    print(csys.jac_obs_x(None, x, None))
    print(csys.jac_obs_theta(None, x, None))


if __name__ == '__main__':
    main()
