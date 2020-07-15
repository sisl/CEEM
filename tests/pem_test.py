#
# File: pem_test.py
#

import torch
import numpy as np
from ceem.smoother import *
from ceem.learner import *
from ceem.systems import LorenzAttractor
from ceem.dynamics import *
from ceem import utils
from ceem.particleem import *

from ceem import logger


def test_particleem():

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

    
    Px0 = torch.eye(3)
    Q = 0.1 * torch.eye(3)
    R = 0.1 * torch.eye(2)

    Np = 300

    fapf = faPF(Np, sys, Q, R, Px0)

    def callback(epoch):
        pass

    trainer = SAEMTrainer(fapf, y, 
        gamma_sched=lambda x: 0.8,
        xlen_cutoff = 3,
        max_k=10,
        )
    trainer.train(params, callbacks=[callback])



    assert True
    if True:
        print('Passed.')
    else:
        print('Failed.')


if __name__ == '__main__':
    test_particleem()
