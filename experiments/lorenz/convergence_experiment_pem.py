import collections
import itertools
import json
import os
import shutil
from copy import deepcopy

import click
import joblib
import numpy as np
import pandas as pd
import torch

from ceem import logger, utils
from ceem.dynamics import *
from ceem.learner import *
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem.smoother import *
from ceem.systems import LorenzSystem, default_lorenz_system

from ceem.particleem import *

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

opj = os.path.join


@click.command()
@click.option('--sys-seed', default=4, type=int)
@click.option('--num-seeds', default=4, type=int)
@click.option('--logdir', default='./data/lorenz/convergence_experiment/pem', type=click.Path())
def run(sys_seed, num_seeds, logdir):

    if os.path.exists(logdir):
        print('Directory exists. Press d to delete.')
        action = None
        while not action:
            action = input().lower().strip()
        act = action
        if act == 'd':
            shutil.rmtree(logdir)
        else:
            quit()
    os.mkdir(logdir)

    k = 6
    Brange = [8] #, 4, 2]

    parallel = True

    if parallel:
        joblib.Parallel(n_jobs=10)(joblib.delayed(train)(seed, opj(logdir, f'k={k}_B={B}_seed={seed}'),
                                                        sys_seed, k=k, b=B)
                              for seed, B in itertools.product(range(42, 42 + num_seeds), Brange))

    else:
        for B in Brange:
            for seed in range(42, 42 + num_seeds):
                logdir_ = opj(logdir, 'k=%d_B=%d_seed=%d' % (k, B, seed))
                train(seed, logdir_, sys_seed, k, B)
    

def plot3d(ax, x, y, z, **kwargs):
    ax.plot(x.detach().numpy(), y.detach().numpy(), z.detach().numpy(), **kwargs)


def train(seed, logdir, sys_seed, k, b):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    ystd = 0.01
    # ystd = 0.

    torch.set_default_dtype(torch.float64)

    logger.setup(logdir, action='d')

    N = 128

    n = 3 * k

    B = b

    true_system = default_lorenz_system(k, obsdif=2)

    utils.set_rng_seed(sys_seed)

    xdim = true_system.xdim
    ydim = true_system.ydim

    dt = true_system._dt

    x0mean = torch.tensor([[-6] * k + [-6] * k + [24.] * k]).unsqueeze(0)

    # simulate true_dynamics over IC distribution
    x_test = x0mean.repeat(1024, 1, 1)
    x_test += 5.0 * torch.randn_like(x_test)
    x_test = x_test.detach()
    t_test = torch.zeros(1024, 1)
    tgt_test = true_system.step_derivs(t_test, x_test).detach()


    Q = (0.01 ** 2) * torch.eye(true_system.xdim)
    R = (ystd ** 2) * torch.eye(true_system.ydim)
    Px0 = 2.5**2 * torch.eye(true_system.xdim)

    ## simulate the true system

    xs = [x0mean.repeat(B, 1, 1)]
    xs[0] += 2.5 * torch.randn_like(xs[0])
    with torch.no_grad():
        for t in range(N - 1):
            xs.append(true_system.step(torch.tensor([0.] * B), xs[-1]))

    xs = torch.cat(xs, dim=1)

    fig = plt.figure()
    for b in range(B):
        ax = fig.add_subplot(int(np.ceil(B / 2.)), 2, b + 1, projection='3d')

        for k_ in range(k):
            plot3d(plt.gca(), xs[b, :, k_], xs[b, :, k + k_], xs[b, :, 2 * k + k_], linestyle='--',
                   alpha=0.5)

    plt.savefig(os.path.join(logger.get_dir(), 'figs/traj_%d.png' % b), dpi=300)
    # plt.show()
    plt.close()

    t = torch.tensor(range(N)).unsqueeze(0).expand(B, -1).to(torch.get_default_dtype())

    y = true_system.observe(t, xs).detach()

    # seed for real now
    utils.set_rng_seed(seed)

    y += ystd * torch.randn_like(y)

    # prep system
    system = deepcopy(true_system)

    true_params = parameters_to_vector(true_system.parameters())

    utils.set_rng_seed(seed)

    params = true_params * ((torch.rand_like(true_params) - 0.5) / 5. + 1.)  # within 10%


    vector_to_parameters(params, system.parameters())

    params = list(system.parameters())

    Np = 100

    fapf = faPF(Np, system, Q, R, Px0)

    timer = {'start_time':timeit.default_timer()}

    def ecb(epoch):

        logger.logkv('time/epoch', epoch)

        params = list(system.parameters())
        vparams = parameters_to_vector(params)

        error = (vparams - true_params).norm().item()

        logger.logkv('test/log10_paramerror', np.log10(error))

        logger.logkv('time/epochtime', timeit.default_timer() - timer['start_time'])

        timer['start_time'] = timeit.default_timer()

        with torch.no_grad():
            tgt_test_pr = system.step_derivs(t_test, x_test)
            error = float(torch.nn.functional.mse_loss(tgt_test_pr, tgt_test))

        logger.logkv('test/log10_error', np.log10(error))

        return

    epoch_callbacks = [ecb]

    ecb(-1)
    logger.dumpkvs()

    trainer = SAEMTrainer(fapf, y, 
        # gamma_sched=lambda x: HarmonicDecayScheduler(x, a=50.),
        gamma_sched=lambda x: 0.2,
        max_k=5000,
        xlen_cutoff = 15,
        )
    trainer.train(params, callbacks=epoch_callbacks)

    return


if __name__ == '__main__':
    run()
