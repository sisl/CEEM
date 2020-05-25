import json
import os
from copy import deepcopy

import click
import numpy as np
import torch
from scipy.io import loadmat

from ceem import logger, utils
from ceem.data_utils import *
from ceem.dynamics import *
from ceem.exp_utils import *
from ceem.learner import *
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem.smoother import *
from ceem.systems import DiscreteLinear
from ceem.systems.discretelinear import DEFAULT_NN

opj = os.path.join


@click.command()
@click.option('--logdir', type=str, default='./data/NLobsLdyn')
@click.option('--datafile', type=click.Path(),
              default='./datasets/split_normalized/train_resid_data.npz')
@click.option('--valdatafile', type=click.Path(),
              default='./datasets/split_normalized/valid_resid_data.npz')
@click.option('--subset', type=int, default=None)
@click.option('--practice', type=int, default=0)
@click.option('--wfac', type=float, default=1.)
@click.option('--xdim', type=int, default=10)
@click.option('--obsmodel', type=int, default=1)
@click.option('--hotstartdir', type=click.Path(), default='./pretrained_models/SID')
@click.option('--seed', type=int, default=1)
@click.option('--parallel', type=int, default=0)
def main(logdir, datafile, valdatafile, subset, obsmodel, practice, wfac, xdim, hotstartdir, seed,
         parallel):
    utils.set_rng_seed(seed)

    torch.set_default_dtype(torch.float64)

    logger.setup(logdir)

    # load train data
    traindata = np.load(datafile)
    u = torch.tensor(traindata['u'])
    y = torch.tensor(traindata['y'])

    valdata = np.load(valdatafile)
    val_u = torch.tensor(valdata['u'])
    val_y = torch.tensor(valdata['y'])

    _, y_std, _, _ = load_statistics('./datasets/split_normalized')

    if practice == 1:
        inds = np.random.choice(466, 10, replace=False)
        u = u[inds]
        y = y[inds]

    B, T, ydim = y.shape
    _, _, udim = u.shape

    t = torch.stack([torch.arange(T)] * B).to(torch.get_default_dtype())

    # specify system
    sid = loadmat(opj(hotstartdir, 'SID_%dD.mat' % xdim))
    A = torch.tensor(sid['A']).to(torch.get_default_dtype())
    Bsys = torch.tensor(sid['B']).to(torch.get_default_dtype())
    C = torch.tensor(sid['C']).to(torch.get_default_dtype())
    D = torch.tensor(sid['D']).to(torch.get_default_dtype())
    NN = deepcopy(DEFAULT_NN)
    NN['gain'] = 0.1
    sys = DiscreteLinear(xdim, udim, ydim, A=A, B=Bsys, C=C, D=D, obsModel=NN)

    # specify smoothing criteria

    smoothing_criteria = []

    vfac = 1.0

    for b in range(B):

        obscrit = GaussianObservationCriterion(vfac * torch.ones(ydim), t[b:b + 1], y[b:b + 1],
                                               u=u[b:b + 1])

        dyncrit = GaussianDynamicsCriterion(wfac * torch.ones(xdim), t[b:b + 1], u=u[b:b + 1])

        smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

    smooth_solver_kwargs = {'verbose': 0, 'tr_rho': 0.1}

    # specify learning criteria
    learning_criteria = [
        GaussianObservationCriterion(torch.ones(ydim), t, y, u=u),
        GaussianDynamicsCriterion(torch.ones(xdim), t, u=u)
    ]
    learning_params = [list(sys._obs.parameters()), list(sys._dyn.parameters())]

    learning_opts = ['torch_minimize', 'torch_minimize']
    learner_opt_kwargs = [{
        'method': 'Adam',
        'lr': 5e-4,
        'nepochs': 500,
        'tr_rho': 0.5
    }, {
        'method': 'Adam',
        'lr': 1e-3,
        'nepochs': 500,
        'tr_rho': 0.5
    }]

    # save params
    run_params = dict(seed=seed, subset=subset, xdim=xdim, vfac=vfac, wfac=wfac,
                      learning_opts=learning_opts, learner_opt_kwargs=learner_opt_kwargs,
                      smooth_solver_kwargs=smooth_solver_kwargs, practice=practice,
                      obsmodel=obsmodel)

    with open(opj(logdir, 'run_params.json'), 'w') as f:
        json.dump(run_params, f)

    # instantiate CEEM

    class Tracker:

        def __init__(self):
            return

    tracker = Tracker()
    tracker.best_val_rmse = np.inf

    def ecb(epoch):
        torch.save(sys.state_dict(),
                   os.path.join(logger.get_dir(), 'ckpts', 'model_{}.th'.format(epoch)))
        y_pred = gen_ypred_model(sys, val_u, val_y)
        rms = compute_rms(val_y[:, 25:], y_pred[:, 25:], y_std)
        val_rmse = float(rms.mean())
        logger.logkv('test/val_rmse', val_rmse)
        if val_rmse < tracker.best_val_rmse:
            tracker.best_val_rmse = val_rmse
            torch.save(sys.state_dict(), os.path.join(logger.get_dir(), 'ckpts', 'best_model.th'))
        return

    epoch_callbacks = [ecb]

    def tcb(epoch):
        # TODO
        return False

    termination_callback = tcb

    ceem = CEEM(smoothing_criteria, learning_criteria, learning_params, learning_opts,
                epoch_callbacks, termination_callback, parallel=parallel)

    # run CEEM

    x0 = 0.01 * torch.randn(B, T, xdim)

    ceem.train(xs=x0, sys=sys, nepochs=5000, smooth_solver_kwargs=smooth_solver_kwargs,
               learner_opt_kwargs=learner_opt_kwargs, subset=subset)


if __name__ == '__main__':
    main()
