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

from copy import deepcopy

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
    ys = torch.cat(ys, dim=1)

    m = ys.shape[-1]


    fig = plt.figure()
    for b in range(B):
        ax = fig.add_subplot(int(np.ceil(B / 2.)), 2, b + 1, projection='3d')

        for k_ in range(k):
            plot3d(plt.gca(), x[b, :, k_], x[b, :, k + k_], x[b, :, 2 * k + k_], linestyle='--',
                   alpha=0.5)

    plt.savefig(opj(logdir, 'traintrajs.png'), dpi=300)

    true_system = deepcopy(sys)

    Np = 100

    params = [sys._sigma, sys._rho, sys._beta]
    true_vparams = parameters_to_vector(params)
    pert_vparams = true_vparams * ((torch.rand_like(true_vparams) - 0.5)/5 + 1.0)
    vector_to_parameters(pert_vparams, params)

    fapf = faPF(Np, sys, Q, R, Px0)

    timer = {'start_time':timeit.default_timer()}

    
    def callback(epoch):
        logger.logkv('test/rho', float(sys._rho))
        logger.logkv('test/sigma', float(sys._sigma))
        logger.logkv('test/beta', float(sys._beta))

        logger.logkv('test/rho_pcterr_log10',
                     float(torch.log10((true_system._rho - sys._rho).abs() / true_system._rho)))
        logger.logkv(
            'test/sigma_pcterr_log10',
            float(torch.log10((true_system._sigma - sys._sigma).abs() / true_system._sigma)))
        logger.logkv(
            'test/beta_pcterr_log10',
            float(torch.log10((true_system._beta - sys._beta).abs() / true_system._beta)))

        logger.logkv('time/epochtime', timeit.default_timer() - timer['start_time'])

        timer['start_time'] = timeit.default_timer()

        return

    callback(-1)
    logger.dumpkvs()

    trainer = SAEMTrainer(fapf, ys, 
        # gamma_sched=lambda x: HarmonicDecayScheduler(x, a=50.),
        gamma_sched=lambda x: 0.8
        )
    trainer.train(params, callbacks=[callback])


if __name__ == '__main__':

    for seed in range(43, 53):
        train(seed, 'data/lorenz/comp/pem/seed%d'%seed, ystd=0.5, wstd=0.1)





