#
# File: ceem_test.py
#

import torch
import numpy as np
from ceem.smoother import *
from ceem.learner import *
from ceem.systems import LorenzAttractor
from ceem.dynamics import *
from ceem.opt_criteria import *
from ceem import utils
from ceem.ceem import CEEM

from ceem import logger


def test_ceem():

    utils.set_rng_seed(1)

    torch.set_default_dtype(torch.float64)

    # setup system
    sigma = torch.tensor([10.])
    rho = torch.tensor([28.])
    beta = torch.tensor([8. / 3.])

    C = torch.randn(2, 3)

    dt = 0.04

    sys = LorenzAttractor(sigma, rho, beta, C, dt, method='midpoint')

    # simulate
    B = 2
    T = 20
    xs = [torch.randn(B, 1, 3)]
    for t in range(T - 1):
        xs.append(sys.step(torch.tensor([0.] * B), xs[-1]))

    xtr = torch.cat(xs, dim=1).detach()
    y = sys.observe(0., xtr).detach()
    # y += torch.rand_like(y) * 0.01

    t = torch.stack([torch.arange(T)] * B).to(torch.get_default_dtype())

    #
    params = list(sys.parameters())
    vparams = parameters_to_vector(params)
    true_vparams = vparams.clone()
    vparams *= 1. + (torch.rand_like(vparams) - 0.5) * 2 * 0.025
    vector_to_parameters(vparams, params)

    # specify smoothing criteria

    smoothing_criteria = []

    for b in range(B):

        obscrit = GaussianObservationCriterion(sys, torch.ones(2), t[b:b + 1], y[b:b + 1])

        dyncrit = GaussianDynamicsCriterion(sys, torch.ones(3), t[b:b + 1])

        smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

    smooth_solver_kwargs = {'verbose':0, 'tr_rho': 0.001}

    # specify learning criteria
    learning_criteria = [[GaussianDynamicsCriterion(sys, torch.ones(3), 
            t[b:b + 1]) for b in range(B)]]
    learning_params = [params]
    learning_opts = ['scipy_minimize']
    learner_opt_kwargs = {'method': 'Nelder-Mead',
                          'tr_rho': 0.01}

    # learning_opts = ['torch_minimize']
    # learner_opt_kwargs = {'method': 'Adam',
    #                       'tr_rho': 0.01}

    # instantiate CEEM

    def ecb(epoch):
        return

    epoch_callbacks = [ecb]

    def tcb(epoch):

        params = list(sys.parameters())
        vparams = parameters_to_vector(params)

        error = (vparams - true_vparams).norm().item()

        logger.logkv('test/log10_error', np.log10(error))

        return error < 5e-3

    termination_callback = tcb

    ceem = CEEM(smoothing_criteria, learning_criteria, learning_params, learning_opts,
                epoch_callbacks, termination_callback) #, parallel=2)

    # run CEEM

    x0 = torch.zeros_like(xtr)

    x0 = [x0[b:b+1] for b in range(B)]

    ceem.train(xs=x0, nepochs=150, 
        smooth_solver_kwargs = smooth_solver_kwargs,
        learner_opt_kwargs=learner_opt_kwargs, subset=1)

    assert tcb(0)
    if tcb(0):
        print('Passed.')
    else:
        print('Failed.')


if __name__ == '__main__':
    test_ceem()
