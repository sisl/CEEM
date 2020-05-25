import torch

from ceem.opt_criteria import *
from ceem.systems import LorenzAttractor
from ceem.dynamics import *
from ceem.smoother import *
from ceem import utils
from ceem.learner import learner

from torch.nn.utils import vector_to_parameters, parameters_to_vector


def test_learner():

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

    t = torch.stack([torch.arange(T), torch.arange(T)]).to(torch.get_default_dtype())

    dyncrit = GaussianDynamicsCriterion(torch.ones(3), t)

    params = list(sys.parameters())[:2]
    vparams = parameters_to_vector(params)
    true_vparams = vparams.clone()
    vparams += torch.randn_like(vparams) * 0.1
    vector_to_parameters(vparams, params)

    opt_result = learner(sys, [dyncrit], [x], ['scipy_minimize'], [params], [{}], opt_kwargs_list=[{
        'method': 'Nelder-Mead'
    }])[0]

    params = list(sys.parameters())[:2]
    vparams = parameters_to_vector(params)

    error = (vparams - true_vparams).norm().item()

    assert np.allclose(error, 0., atol=1e-4), 'Error=%.3e' % error

    print('Passed.')


if __name__ == '__main__':
    test_learner()
