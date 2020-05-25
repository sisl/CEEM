from collections import OrderedDict

import torch
from torch import nn

from ceem.dynamics import AnalyticObsJacMixin, C2DSystem, DynJacMixin
from ceem.utils import set_rng_seed


class LorenzSystem(C2DSystem, nn.Module, AnalyticObsJacMixin, DynJacMixin):
    """Basic Lorenz Attractor
    """

    def __init__(self, sigmas, rhos, betas, H, C, dt, method='midpoint'):
        """
           Args:
              sigma (torch.tensor): (n//3,) torch tensor 
              rho (torch.tensor): (n//3,) torch tensor
              beta (torch.tensor): (n//3,) torch tensor
              H (torch.tensor): (n,n) coupling matrix
              C (torch.tensor): (ydim, n) observation matrix
        """

        C2DSystem.__init__(self, dt=dt, method=method)
        nn.Module.__init__(self)

        n = H.shape[0]
        assert n % 3 == 0, 'n must be a multiple of 3'

        self._n = n

        self._sigmas = nn.Parameter(sigmas.unsqueeze(0).unsqueeze(0))
        self._rhos = nn.Parameter(rhos.unsqueeze(0).unsqueeze(0))
        self._betas = nn.Parameter(betas.unsqueeze(0).unsqueeze(0))
        self._H = nn.Parameter(H.unsqueeze(0).unsqueeze(0))

        self._C = C.unsqueeze(0).unsqueeze(0)  # not a learned parameter

        self._xdim = n
        self._udim = None
        self._ydim = C.shape[0]

    def step_derivs(self, t, x, u=None):

        n_ = self._n // 3

        x_ = x[:, :, :n_]
        y_ = x[:, :, n_:2 * n_]
        z_ = x[:, :, 2 * n_:]

        dxdt = self._sigmas * (y_ - x_)
        dydt = x_ * (self._rhos - z_) - y_
        dzdt = x_ * y_ - self._betas * z_

        dinpdt = torch.cat([dxdt, dydt, dzdt], dim=2)

        # coupling
        dinpdt = dinpdt + (self._H @ x.unsqueeze(3)).squeeze(3)

        return dinpdt

    def observe(self, t, x, u=None):

        return (self._C @ x.unsqueeze(3)).squeeze(3)

    def jac_obs_x(self, t, x, u=None):
        B, T, n = x.shape
        return self._C.repeat(B, T, 1, 1)

    def jac_obs_theta(self, t, x, u=None):
        # observation doesnt depend on learned params
        jacobians = OrderedDict([
            ('_sigma', None),
            ('_rho', None),
            ('_beta', None),
        ])
        return jacobians


def default_lorenz_system(k, seed=4, obsdif=2, dt=0.04):

    currng = torch.random.get_rng_state()

    torch.manual_seed(seed)

    n = 3 * k

    sigmas = torch.tensor([10.] * k)
    rhos = torch.tensor([28.] * k)
    betas = torch.tensor([8. / 3] * k)

    # generate coupling matrix
    H = torch.randn(n, n)
    for i in range(k):
        H[i::k, i::k] = 0.

    H = 5. * H / H.norm()

    obsdim = n - obsdif
    C = torch.randn(obsdim, n)

    true_system = LorenzSystem(sigmas, rhos, betas, H, C, dt, method='rk4')

    torch.random.set_rng_state(currng)

    return true_system


if __name__ == '__main__':
    main()
