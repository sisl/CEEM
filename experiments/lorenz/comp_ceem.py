import torch
from ceem import utils
from torch.distributions.categorical import Categorical

from torch.distributions.multivariate_normal import MultivariateNormal
from ceem.particleem import * 

from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ceem.systems import LorenzAttractor, default_lorenz_attractor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import os

import collections
import json
import os
import shutil
import tempfile
from copy import deepcopy

import click
import numpy as np
import pandas as pd

import torch
from ceem import logger, utils
from ceem.dynamics import *
from ceem.learner import *
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem.smoother import *
from ceem.systems import LorenzAttractor, default_lorenz_attractor


import timeit

opj = os.path.join

torch.set_default_dtype(torch.float64)

def plot3d(ax, x, y, z, **kwargs):
    ax.plot(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), **kwargs)


def train(seed, logdir, ystd=0.1, wstd=0.01, sys_seed=4):

    print('\n\n\n##### SEED %d #####\n\n'%seed)

    torch.set_default_dtype(torch.float64)

    logger.setup(logdir, action='d')
    
    # Number of timesteps in the trajectory
    T = 128

    n = 3

    # Batch size
    B = 4

    k = 1

    utils.set_rng_seed(sys_seed)

    sys = default_lorenz_attractor()

    dt = sys._dt

    utils.set_rng_seed(seed)

    # simulate the system

    x0mean = torch.tensor([[-6] * k + [-6] * k + [24.] * k])
    x0mean = x0mean.unsqueeze(0).repeat(B,1,1)

    # Rollout with noise

    Q = (wstd ** 2) * torch.eye(sys.xdim)
    R = (ystd ** 2) * torch.eye(sys.ydim)
    Px0 = 5.0 * torch.eye(sys.xdim)

    Qpdf = MultivariateNormal(torch.zeros((B,1,sys.xdim)), Q.unsqueeze(0).unsqueeze(0))
    Rpdf = MultivariateNormal(torch.zeros((B,1,sys.ydim)), R.unsqueeze(0).unsqueeze(0))
    Px0pdf = MultivariateNormal(x0mean, Px0.unsqueeze(0).unsqueeze(0))


    xs = [Px0pdf.sample()]
    ys = [sys.observe(0, xs[0]) + Rpdf.sample()]

    for t in range(T-1):

        tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
        xs.append(sys.step(tinp, xs[-1]) + Qpdf.sample())
        ys.append(sys.observe(tinp, xs[-1]) + Rpdf.sample())

    x = torch.cat(xs, dim=1)
    y = torch.cat(ys, dim=1)


    t = torch.tensor(range(T)).unsqueeze(0).to(torch.get_default_dtype()).repeat(B,1)

    m = y.shape[-1]


    fig = plt.figure()
    for b in range(B):
        ax = fig.add_subplot(int(np.ceil(B / 2.)), 2, b + 1, projection='3d')

        for k_ in range(k):
            plot3d(plt.gca(), x[b, :, k_], x[b, :, k + k_], x[b, :, 2 * k + k_], linestyle='--',
                   alpha=0.5)

    plt.savefig(opj(logdir, 'traintrajs.png'), dpi=300)

    
    # prep system
    true_system = sys

    system = deepcopy(true_system)

    true_params = parameters_to_vector(true_system.parameters())

    params = true_params * ((torch.rand_like(true_params) - 0.5) / 5. + 1.)  # within 10%

    vector_to_parameters(params, system.parameters())

    params = list(system.parameters())

    # specify smoothing criteria

    smoothing_criteria = []

    for b in range(B):

        obscrit = GaussianObservationCriterion(system, torch.ones(2), t[b:b + 1], y[b:b + 1])

        dyncrit = GaussianDynamicsCriterion(system, wstd / ystd * torch.ones(3), t[b:b + 1])

        smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

    smooth_solver_kwargs = {'verbose': 0, 'tr_rho': 0.001}

    # specify learning criteria
    learning_criteria = [GaussianDynamicsCriterion(system, torch.ones(3), t)]
    learning_params = [params]
    learning_opts = ['scipy_minimize']
    learner_opt_kwargs = {'method': 'Nelder-Mead', 'tr_rho': 0.01}

    # instantiate CEEM



    timer = {'start_time':timeit.default_timer()}

    def ecb(epoch):
        logger.logkv('test/rho', float(system._rho))
        logger.logkv('test/sigma', float(system._sigma))
        logger.logkv('test/beta', float(system._beta))

        logger.logkv('test/rho_pcterr_log10',
                     float(torch.log10((true_system._rho - system._rho).abs() / true_system._rho)))
        logger.logkv(
            'test/sigma_pcterr_log10',
            float(torch.log10((true_system._sigma - system._sigma).abs() / true_system._sigma)))
        logger.logkv(
            'test/beta_pcterr_log10',
            float(torch.log10((true_system._beta - system._beta).abs() / true_system._beta)))


        logger.logkv('time/epochtime', timeit.default_timer() - timer['start_time'])

        timer['start_time'] = timeit.default_timer()

        return

    epoch_callbacks = [ecb]

    class Last10Errors:

        def __init__(self):
            return

    last_10_errors = Last10Errors
    last_10_errors._arr = []

    def tcb(epoch):

        params = list(system.parameters())
        vparams = parameters_to_vector(params)

        error = (vparams - true_params).norm().item()

        last_10_errors._arr.append(float(error))

        logger.logkv('test/log10_error', np.log10(error))

        if len(last_10_errors._arr) > 10:
            last_10_errors._arr = last_10_errors._arr[-10:]

            l10err = torch.tensor(last_10_errors._arr)

            convcrit = float((l10err.min() - l10err.max()).abs())
            logger.logkv('test/log10_convcrit', np.log10(convcrit))
            if convcrit < 1e-4:
                return True

        return False

    termination_callback = tcb

    ceem = CEEM(smoothing_criteria, learning_criteria, learning_params, learning_opts,
                epoch_callbacks, termination_callback)

    # run CEEM

    x0 = torch.zeros_like(x)

    ecb(-1)
    logger.dumpkvs()

    ceem.train(xs=x0, nepochs=100, smooth_solver_kwargs=smooth_solver_kwargs,
               learner_opt_kwargs=learner_opt_kwargs)

    return float(system._sigma), float(system._rho), float(system._beta)


if __name__ == '__main__':

    for seed in range(43, 43+10):
        train(seed, 'data/lorenz/comp/ceem/seed%d'%seed, ystd=0.5, wstd=0.1)





