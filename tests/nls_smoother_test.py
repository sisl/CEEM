import torch

from ceem.opt_criteria import *
from ceem.systems import LorenzAttractor
from ceem.dynamics import *
from ceem.smoother import *
from ceem import utils


def test_smoother():

    utils.set_rng_seed(1)

    torch.set_default_dtype(torch.float64)

    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    dt = 0.04

    sys = LorenzAttractor(sigma, rho, beta, C, dt, method='midpoint')

    B = 1
    T = 200
    xs = [torch.randn(B, 1, 3)]
    for t in range(T - 1):
        xs.append(sys.step(torch.tensor([0.] * B), xs[-1]))

    x = torch.cat(xs, dim=1).detach()
    x.requires_grad = True
    y = sys.observe(0., x).detach()
    # y += torch.rand_like(y) * 0.01

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())
    
    x0 = torch.zeros_like(x)

    obscrit = GaussianObservationCriterion(sys, torch.ones(2), t, y)

    dyncrit = GaussianDynamicsCriterion(sys, torch.ones(3), t)

    # Test GroupSOSCriterion
    crit = GroupSOSCriterion([obscrit, dyncrit])

    xsm, metrics = NLSsmoother(x0, crit, solver_kwargs={'verbose': 2, 'tr_rho': 0.})

    err = float((xsm - x).norm())
    assert err < 1e-8, 'Smoothing Error: %.3e' % err

    print('Passed.')

    # Test BlockSparseGroupSOSCriterion
    crit = BlockSparseGroupSOSCriterion([obscrit, dyncrit])

    xsm, metrics = NLSsmoother(torch.zeros_like(x), crit)

    err = float((xsm - x).norm())
    assert err < 1e-8, 'Smoothing Error: %.3e' % err

    print('Passed.')

if __name__ == '__main__':
    test_smoother()
