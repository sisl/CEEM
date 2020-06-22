#
# File: criterion_test.py
#

from ceem.opt_criteria import *
from ceem.systems import LorenzAttractor, SpringMassDamper, DiscreteLinear
from ceem.dynamics import *
from ceem import utils

from scipy.sparse.linalg import norm

import numpy as np


def check_sys(sys, t, x, y, atol=1e-8, u=None):
    B, T, n = x.shape
    _, _, m = y.shape

    # GaussianObservationCriterion
    obscrit = GaussianObservationCriterion(sys, 0.5 * torch.eye(m), t, y, u=u)

    obsjac = obscrit.jac_resid_x(x)

    obsjac_slow = super(GaussianObservationCriterion, obscrit).jac_resid_x(x)

    test = float((obsjac - obsjac_slow).norm())
    assert np.allclose(test, 0., atol=atol), 'ObsJac Torch Comparison: %.3e' % test

    obsjac = obscrit.jac_resid_x(x, sparse=True)

    obsjac_slow = super(GaussianObservationCriterion, obscrit).jac_resid_x(x, sparse=True)

    test = norm(obsjac - obsjac_slow)
    assert np.allclose(test, 0., atol=atol), 'ObsJac Sparse Comparison: %.3e' % test

    # SoftTrustRegionCriterion
    trcrit = STRStateCriterion(rho=2., x0=x.clone())

    trjac = trcrit.jac_resid_x(x)

    trjac_slow = super(STRStateCriterion, trcrit).jac_resid_x(x)
    test = float((trjac - trjac_slow).norm())
    assert np.allclose(test, 0., atol=atol), 'trJac Torch Comparison: %.3e' % test

    trjac = trcrit.jac_resid_x(x, sparse=True)

    trjac_slow = super(STRStateCriterion, trcrit).jac_resid_x(x, sparse=True)

    test = norm(trjac - trjac_slow)
    assert np.allclose(test, 0., atol=atol), 'trJac Sparse Comparison: %.3e' % test

    # GaussianDynamicsCriterion
    dyncrit = GaussianDynamicsCriterion(sys, 0.75 * torch.ones(n), t, u=u)

    dynjac = dyncrit.jac_resid_x(x)

    dynjac_slow = super(GaussianDynamicsCriterion, dyncrit).jac_resid_x(x)

    test = float((dynjac - dynjac_slow).norm())
    assert np.allclose(test, 0., atol=atol), 'DynJac Torch Comparison: %.3e' % test

    dynjac = dyncrit.jac_resid_x(x, sparse=True)

    dynjac_slow = super(GaussianDynamicsCriterion, dyncrit).jac_resid_x(x, sparse=True)

    test = norm(dynjac - dynjac_slow)
    assert np.allclose(test, 0., atol=atol), 'DynJac Sparse Comparison: %.3e' % test

    #GroupSOSCriterion
    groupcrit = GroupSOSCriterion([trcrit, obscrit, dyncrit])

    groupjac = groupcrit.jac_resid_x(x)

    groupjac_slow = super(GroupSOSCriterion, groupcrit).jac_resid_x(x)

    test = float((groupjac - groupjac_slow).norm())
    assert np.allclose(test, 0., atol=atol), 'GroupJac Torch Comparison: %.3e' % test

    groupjac = groupcrit.jac_resid_x(x, sparse=True)

    groupjac_slow = super(GroupSOSCriterion, groupcrit).jac_resid_x(x, sparse=True)

    test = norm(groupjac - groupjac_slow)
    assert np.allclose(test, 0., atol=atol), 'GroupJac Sparse Comparison: %.3e' % test

    # BlockSparseGroupSOSCriterion
    groupcrit = BlockSparseGroupSOSCriterion([trcrit, obscrit, dyncrit])

    groupjac = groupcrit.jac_resid_x(x, )

    groupjac_slow = super(BlockSparseGroupSOSCriterion, groupcrit).jac_resid_x(x, sparse=True)

    test = norm(groupjac - groupjac_slow)
    assert np.allclose(test, 0., atol=atol), 'BlockSparseGroupJac Comparison: %.3e' % test

    print('Passed.')


def test_sys():

    utils.set_rng_seed(1)
    torch.set_default_dtype(torch.float64)

    # test LorenzAttractor

    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    dt = 0.04

    sys = LorenzAttractor(sigma, rho, beta, C, dt, method='midpoint')


    B = 5
    T = 20
    xs = [torch.randn(B, 1, 3)]
    for t in range(T - 1):
        xs.append(sys.step(torch.tensor([0.] * B), xs[-1]))

    x = torch.cat(xs, dim=1).detach()
    x.requires_grad = True
    y = sys.observe(0., x)
    y += torch.rand_like(y) * 0.01

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())

    check_sys(sys, t, x, y)

    # test SpringMassDamper

    n = 4

    M = D = K = torch.tensor([[1., 2.], [2., 5.]])

    dt = 0.1

    method = 'midpoint'

    sys = SpringMassDamper(M, D, K, dt, method=method)

    B = 5
    T = 20
    xs = [torch.randn(B, 1, n)]
    for t in range(T - 1):
        xs.append(sys.step(torch.tensor([0.] * B), xs[-1]))

    x = torch.cat(xs, dim=1).detach()
    x.requires_grad = True
    y = sys.observe(0., x).detach()
    y += torch.rand_like(y) * 0.01

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())

    check_sys(sys, t, x, y)

    # test DiscreteLinear

    xdim = 2
    ydim = 3
    udim = 2

    sys = DiscreteLinear(xdim, udim, ydim)

    x = torch.randn(B,T,xdim)
    u = torch.randn(B,T,udim)
    y = torch.randn(B,T,ydim)

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())    

    check_sys(sys,t,x,y,u=u)

if __name__ == '__main__':
    test_sys()
