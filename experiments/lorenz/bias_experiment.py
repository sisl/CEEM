#
# bias_experiment.py
#
# Experiment in  Paper's Section 3.1.1
#

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


@click.command()
@click.option('--sys-seed', default=4, type=int)
@click.option('--num-seeds', default=10, type=int)
@click.option('--logdir', default='./data/bias_experiment', type=click.Path())
def run(sys_seed, num_seeds, logdir):
    # Delete old version
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.mkdir(logdir)

    results = collections.defaultdict(list)

    ystd = 1e-2
    for wstd in [1e-1, 1e-2, 1e-3]:
        for seed in range(num_seeds):
            tmpdir = tempfile.mkdtemp()
            sigma, rho, beta = train(seed, tmpdir, sys_seed, ystd, wstd)

            results['ystd'].append(ystd)
            results['wstd'].append(wstd)
            results['seed'].append(seed)
            results['sigma'].append(sigma)
            results['rho'].append(rho)
            results['beta'].append(beta)
        df = pd.DataFrame(results)
        df.to_pickle(os.path.join(logdir, 'results.pkl'))

    wstd = 1e-3
    for ystd in [1e-1, 5e-2]:
        for seed in range(num_seeds):
            tmpdir = tempfile.mkdtemp()
            sigma, rho, beta = train(seed, tmpdir, sys_seed, ystd, wstd)

            results['ystd'].append(ystd)
            results['wstd'].append(wstd)
            results['seed'].append(seed)
            results['sigma'].append(sigma)
            results['rho'].append(rho)
            results['beta'].append(beta)

        df = pd.DataFrame(results)
        df.to_pickle(os.path.join(logdir, 'results.pkl'))

    df = pd.DataFrame(results)
    df.to_pickle(os.path.join(logdir, 'results.pkl'))


def train(seed, logdir, sys_seed, ystd, wstd):

    torch.set_default_dtype(torch.float64)

    logger.setup(logdir, action='d')

    # Number of timesteps in the trajectory
    T = 128

    n = 3

    # Batch size
    B = 1

    k = 1

    utils.set_rng_seed(sys_seed)

    true_system = default_lorenz_attractor()

    dt = true_system._dt

    utils.set_rng_seed(43)

    # simulate the system

    x0mean = torch.tensor([[-6] * k + [-6] * k + [24.] * k]).unsqueeze(0)
    # seed for real now
    utils.set_rng_seed(seed)

    # Rollout with noise
    xs = [x0mean]
    xs[0] += 5. * torch.randn_like(xs[0])
    with torch.no_grad():
        for t in range(T - 1):
            xs.append(
                true_system.step(torch.tensor([0.] * B), xs[-1]) + wstd * torch.randn_like(xs[-1]))

    xs = torch.cat(xs, dim=1)

    t = torch.tensor(range(T)).unsqueeze(0).to(torch.get_default_dtype())

    y = true_system.observe(t, xs).detach()
    y += ystd * torch.randn_like(y)  # Observation noise

    # prep system
    system = deepcopy(true_system)

    true_params = parameters_to_vector(true_system.parameters())

    params = true_params * ((torch.rand_like(true_params) - 0.5) / 5. + 1.)  # within 10%

    vector_to_parameters(params, system.parameters())

    params = list(system.parameters())

    # specify smoothing criteria

    B = 1

    smoothing_criteria = []

    for b in range(B):

        obscrit = GaussianObservationCriterion(torch.ones(2), t[b:b + 1], y[b:b + 1])

        dyncrit = GaussianDynamicsCriterion(wstd / ystd * torch.ones(3), t[b:b + 1])

        smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

    smooth_solver_kwargs = {'verbose': 0, 'tr_rho': 0.001}

    # specify learning criteria
    learning_criteria = [GaussianDynamicsCriterion(torch.ones(3), t)]
    learning_params = [params]
    learning_opts = ['scipy_minimize']
    learner_opt_kwargs = {'method': 'Nelder-Mead', 'tr_rho': 0.01}

    # instantiate CEEM

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

    x0 = torch.zeros_like(xs)

    ceem.train(xs=x0, sys=system, nepochs=500, smooth_solver_kwargs=smooth_solver_kwargs,
               learner_opt_kwargs=learner_opt_kwargs)

    return float(system._sigma), float(system._rho), float(system._beta)


if __name__ == '__main__':
    run()
