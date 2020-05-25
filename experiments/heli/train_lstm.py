#!/usr/bin/env python3
#
# File: train_heli_lstm.py
#
import os
import time

import click
import numpy as np
import torch

from ceem import logger, utils
from ceem.data_utils import load_helidata, load_statistics
from ceem.exp_utils import *

torch.set_default_dtype(torch.float64)
opj = os.path.join

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_LSTM_KWARGS = dict(
    input_size=10,
    hidden_size=32,
    num_layers=2,
    output_size=6,)


class LSTMModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)
        c_0 = torch.zeros(self._num_layers, batch_size, self._hidden_size)
        return (h_0, c_0)

    def forward(self, x):
        B, T, d = x.size()
        h_0, c_0 = self.init_hidden(x.size(0))

        out, (h_out, _) = self.lstm(x, (h_0, c_0))
        out = self.fc(out.reshape(-1, self._hidden_size))
        return out.reshape(B, T, -1)


def train(datadir, logdir, lr, n_epochs, H, save_file='lstm.th'):
    utils.set_rng_seed(1)
    y_mean, y_std, u_mean, u_std = load_statistics(datadir)
    train_data, train_trgt = load_helidata(datadir, 'train')
    test_data, test_trgt = load_helidata(datadir, 'test')
    valid_data, valid_trgt = load_helidata(datadir, 'valid')

    net = LSTMModel(**DEFAULT_LSTM_KWARGS)
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
        for t in range(T - H - 1):
            opt.zero_grad()

            u = train_data[:, t:t + H]
            y = train_trgt[:, t + 1:t + H + 1]
            y_pred = net(u)
            assert y.size() == y_pred.size(), (y_pred.size(), y.size())
            loss = compute_rms(y, y_pred, y_std).mean(0)
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
                    y_pred = net(u)[:, -1, :]
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
                    y_pred = net(u)[:, -1, :]
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
@click.option('--datadir', type=click.Path(), default='./datasets/split_normalized')
@click.option('--logdir', type=click.Path(), default='./data/heli_lstm')
@click.option('--lr', type=float, default=1e-4)
@click.option('--n-epochs', type=int, default=1000)
def run(datadir, logdir, lr, n_epochs):
    train(datadir, logdir, lr, n_epochs, H=25)


if __name__ == '__main__':
    run()
