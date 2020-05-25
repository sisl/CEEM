#
# File: lorenzattractor_test.py
#
import torch
from ceem.systems.lorenzattractor import LorenzAttractor


def test_jacobians():
    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])
    B = 2
    T = 2
    n = 3

    C = torch.randn(2, 3)

    csys = LorenzAttractor(sigma, rho, beta, C, 0.1)
    theta_keys = [k for k, v in csys.named_parameters()]
    x = torch.randn(B, T, n)

    # jac_dyn_theta = csys.jac_dyn_theta(None, x, None)
    # # Test keys length and names
    # assert len(theta_keys) == len(jac_dyn_theta.keys())
    # assert (list(theta_keys) == list(jac_dyn_theta.keys()))

    # for k, v in csys.named_parameters():
    #     assert jac_dyn_theta[k].shape == (B, T, n, 1)

    jac_obs_theta = csys.jac_obs_theta(None, x, None)
    # Test keys length and names
    assert len(theta_keys) == len(jac_obs_theta.keys())
    assert (list(theta_keys) == list(jac_obs_theta.keys()))

    for v in jac_obs_theta.values():
        assert v is None
