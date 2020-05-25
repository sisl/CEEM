import torch
from torch import nn
from ceem.dynamics import DiscreteDynamicalSystem, AnalyticDynJacMixin, AnalyticObsJacMixin, DynJacMixin, ObsJacMixin
from ceem.nn import LNMLP
import numpy as np
from ceem.smoother import EKF

DEFAULT_NN = dict(hidden_sizes=[32]*2, activation='tanh', gain=0.5, ln=False)


class nlsys(DiscreteDynamicalSystem, nn.Module, DynJacMixin, ObsJacMixin):
    """Discrete dynamical system with linear dynamics and linear or non-linear observation model.
    """

    def __init__(self, xdim, udim, ydim):

            super().__init__()
            self._xdim = xdim
            self._ydim = ydim
            self._udim = udim

            self._obs = LNMLP(input_size=udim + xdim, output_size=ydim, **DEFAULT_NN)
            self._dyn = LNMLP(input_size=udim + xdim, output_size=xdim, **DEFAULT_NN)

            _, stdict, _, _ = torch.load('ekf_test_data.pt')
            self.load_state_dict(stdict)

    def step(self, t, x, u):
        return self._dyn(torch.cat([x,u], dim=-1))

    def observe(self, t, x, u):
        return self._obs(torch.cat([x,u], dim=-1))


if __name__ == "__main__":
    xdim = 2
    udim = 2
    ydim = 1

    sigq = 10**(0)
    sigr = 10**(-1)
    system = nlsys(xdim=xdim, udim=udim, ydim=ydim)
    T = 100
    B = 2
    _x, _ , x, y = torch.load('ekf_test_data.pt')
    u = torch.randn((B, T, udim))*0.

    # Run EKF
    Q = torch.eye(xdim) * sigq**2
    R = torch.eye(ydim) * sigr**2

    x0 = torch.tensor([1.,0])
    x_ = EKF(x0, y, u, torch.eye(xdim), Q, R, system)

    test = float((_x[0] - x_[0]).norm() + (_x[1] - x_[1]).norm())
    assert np.allclose(test, 0., atol=1e-8), 'EKF Comparison: %.3e' % test

    print('Passed.')