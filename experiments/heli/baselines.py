import os
import time

import click
import numpy as np
import torch

from ceem import logger, utils
from ceem.baseline_utils import LagModel
from ceem.data_utils import load_helidata, load_statistics
from ceem.exp_utils import *
from ceem.nn import LNMLP

torch.set_default_dtype(torch.float64)

opj = os.path.join

device = 'cuda' if torch.cuda.is_available() else 'cpu'

hyperparameters = {
    1: dict(lr=1e-4, n_epochs=5000, save_file='naive.th', logdir='./data/naive'),
    25: dict(lr=1e-4, n_epochs=5000, save_file='h25.th', logdir='./data/H25')
}


def train_horizon_model(H, datadir):

    # extract hyperparameters
    hp = hyperparameters[H]
    lr = hp['lr']
    n_epochs = hp['n_epochs']
    save_file = hp['save_file']
    logdir = hp['logdir']

    # load data
    utils.set_rng_seed(1)
    y_mean, y_std, u_mean, u_std = load_statistics(datadir)
    train_data, train_trgt = load_helidata(datadir, 'train')
    test_data, test_trgt = load_helidata(datadir, 'test')
    valid_data, valid_trgt = load_helidata(datadir, 'valid')

    #  Define net
    neural_net_kwargs = dict(input_size=10 * H, hidden_sizes=[32] * 8, output_size=6,
                             activation='tanh', gain=1.0, ln=False)

    net = LagModel(neural_net_kwargs, H)

    logger.setup(logdir, action='d')

    # Train
    system = train_net(net, train_data, train_trgt, test_data, test_trgt, valid_data, valid_trgt,
                       y_std, lr, logdir=logdir, H=H, n_epochs=n_epochs)
    torch.save(system.state_dict(), save_file)


def train_net(net, train_data, train_trgt, test_data, test_trgt, valid_data, valid_trgt, y_std, lr,
              H, logdir, n_epochs=1000):

    T = train_data.shape[1]

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler_off = 1000.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda epoch: scheduler_off / (scheduler_off + epoch))

    best_val_loss = np.inf

    t0 = time.time()
    for e in range(n_epochs):
        logger.logkv('epoch', e)
        # train
        ll = []
        coord_error = 0
        for t in range(T - H + 1):
            opt.zero_grad()

            u = train_data[:, t:t + H]
            y = train_trgt[:, t + H - 1]
            y_pred = net(u)
            assert y.size() == y_pred.size()
            loss = compute_rms(y.unsqueeze(0), y_pred.unsqueeze(0), y_std)
            loss.backward()
            opt.step()
            ll.append(float(loss))
        mean_train_loss = np.mean(ll)
        logger.logkv('log10_train_loss', np.log10(mean_train_loss))
        coord_error /= (T - H)

        scheduler.step()

        for param_group in opt.param_groups:
            logger.logkv('log10_lr', np.log10(param_group['lr']))

        if e % 100 == 0:
            # validation
            ll = []
            coord_error = 0
            for t in range(T - H):
                with torch.no_grad():
                    u = valid_data[:, t:t + H]
                    y = valid_trgt[:, t + H - 1]
                    y_pred = net(u)
                    loss = compute_rms(y.unsqueeze(0), y_pred.unsqueeze(0), y_std)
                    ll.append(float(loss))
            mean_val_loss = np.mean(ll)
            logger.logkv('log10_val_loss', np.log10(mean_val_loss))
            coord_error /= (T - H)

            # Test
            ll = []
            coord_error = 0
            for t in range(T - H):
                with torch.no_grad():
                    u = test_data[:, t:t + H]
                    y = test_trgt[:, t + H - 1]
                    y_pred = net(u)
                    loss = compute_rms(y.unsqueeze(0), y_pred.unsqueeze(0), y_std)
                ll.append(float(loss))
            mean_test_loss = np.mean(ll)
            logger.logkv('log10_test_loss', np.log10(mean_test_loss))

            # Save
            if mean_val_loss < best_val_loss:
                torch.save(net.state_dict(), os.path.join(logdir, 'best_net.th'))
                best_val_loss = mean_val_loss

        if time.time() - t0 > 2:
            t0 = time.time()
            logger.dumpkvs()

    return net


@click.command()
@click.option('--model', type=str, default='naive')
@click.option('--datadir', type=click.Path(), default='./datasets/split_normalized')
def run(model, datadir):
    if model == 'naive':
        train_horizon_model(1, datadir)
    elif model == 'H25':
        train_horizon_model(25, datadir)
    else:
        raise ValueError(f"No baseline as {model}")


if __name__ == "__main__":
    run()
