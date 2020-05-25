import matplotlib.pyplot as plt
import torch
import numpy as np
from ceem.data_utils import *
from ceem.smoother import EKF
import pandas as pd
import click
import matplotlib

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}", r"\usepackage{siunitx}"]
ttl = [
    '$a_x$ \n $(\si{\meter\per\second\squared})$', '$a_y$ \n $(\si{\meter\per\second\squared})$', '$a_z$ \n $(\si{\meter\per\second\squared})$',
    '$\dot{\omega}_x$ \n $(\si{\meter\per\second\squared})$', '$\dot{\omega}_y$ \n $(\si{\meter\per\second\squared})$',
    '$\dot{\omega}_z$ \n $(\si{\meter\per\second\squared})$'
]
figsizes = {'large': (10, 4), 'small': (6.4, 4.8)}


@click.command()
@click.option('-b', '--trajectory', type=int, default=9)
@click.option('--datadir', type=click.Path(), default='./datasets/split_normalized')
@click.option('--modelfile', type=click.Path(), default='./experiments/heli/trajectories')
@click.option('-m', '--moments', is_flag=True)
@click.option('-s', '--savename', type=str, default=None)
@click.option('--figsize', type=str, default='large')
def main(trajectory, datadir, modelfile, moments, savename, figsize):
    # load test data
    test_u, test_y, demos = load_helidata(datadir, 'test', return_files=True)
    y_mean, y_std, u_mean, u_std = load_statistics(datadir)
    test_u = test_u * u_std + u_mean
    test_y = test_y * y_std + y_mean
    dt = 0.01
    T = torch.arange(test_y.shape[1], dtype=torch.float32) * dt

    # load predictions
    naivepred = torch.load(f'{modelfile}/naivepred')
    h25pred = torch.load(f'{modelfile}/h25pred')
    sidpred = torch.load(f'{modelfile}/sidpred')
    nlpred = torch.load(f'{modelfile}/nlpred')

    # create plot
    f, ax = plt.subplots(3, 1, figsize=figsizes[figsize])
    b = trajectory

    i = 0
    lines = []
    c = 3 if moments else 0
    for j in range(3):

        lines.append(ax[i].plot(T, test_y[b, :, j + c], alpha=0.8)[0])
        lines.append(ax[i].plot(T[25:], h25pred[b, 1:, j + c], '--', alpha=0.8)[0])
        lines.append(ax[i].plot(T[25:], nlpred[b, 25:, j + c], '--', alpha=0.8)[0])
        lines.append(ax[i].plot(T[25:], sidpred[b, 25:, j + c], '--', alpha=0.8)[0])

        ax[i].set_ylabel(ttl[j + c], rotation=0, ha='center', fontweight='bold', labelpad=20)
        ax[i].grid(True)
        i += 1
    ax[i - 1].set_xlabel('time (s)', fontweight='bold', labelpad=-5)

    lgd = plt.figlegend(handles=lines[:4], labels=['dataset', 'H25', 'NL (ours)', 'SID'],
                        loc='upper center', shadow=True, ncol=4)

    f.subplots_adjust(bottom=0.1)
    plt.tight_layout(rect=[0, 0., 1., .935])

    if savename is None:
        plt.show()
    else:
        plt.savefig(f'./experiments/heli/plotting/{savename}.pdf', bbox_extra_artists=(lgd,),
                    bbox_inches='tight', dpi=400)


if __name__ == "__main__":
    main()
