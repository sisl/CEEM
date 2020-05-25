import os

import click
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from ceem.exp_utils import *

opj = os.path.join


def remove_num(f):
    f = f.split('_')[:-1]
    out = ''
    for i in f:
        out += i
        out += '_'
    return out[:-1]


@click.command()
@click.option('--datadir', type=click.Path(), default='./datasets/split_normalized')
@click.option('--evalfile', type=click.Path(), default='./experiments/heli/trajectories/evaluations.pkl')
@click.option('-s', '--savename', type=str, default='EvalResults')
@click.option('--figsize', type=str, default='large')
def run(datadir, evalfile, savename, figsize):
    _, _, alldemos = load_helidata(datadir, 'test', return_files=True)

    # find unique demos
    demos = set([remove_num(f) for f in alldemos])

    # load evaluate_model.py results
    alldf = pd.read_pickle(evalfile)

    # compute mean rms for each demo
    df = None
    for demo in demos:
        df_ = alldf.loc[alldf['demos'].str.match(demo)]
        df_ = df_.mean()
        df_['demo'] = demo
        df_ = df_.to_dict()
        df_ = pd.DataFrame(df_, index=[0])
        if df is None:
            df = df_
        else:
            df = pd.concat([df, df_], ignore_index=True)

    def plot(fig, name, large=False):
        barwidth = 0.2
        ndemos = len(demos)
        sep = 12 * barwidth

        models = ['naive', 'H25', 'NL', 'lstm']
        labels = ['Naive', 'H25', 'NL (ours)', 'LSTM', 'SID']

        demos_ = df['demo']
        demos_ = ['{}'.format(demo) for demo in demos]

        for i, model in enumerate(models):
            if not large and i==0:
                continue

            color = 'C%i'%i
            if model == 'lstm':
                color = 'C%i'%(i+1)
            ylocs = sep * np.arange(ndemos) + barwidth * i * 2
            plt.barh(ylocs, df[model].values, 2 * barwidth, align='center', label=labels[i],
                     alpha=0.7, color=color)

        i = len(models)
        # plot SID10D mean and std
        cols = [col for col in df.columns if 'SID10D' in col]
        siddf = df[cols].values
        mean_sid = siddf.mean(axis=1)
        std_sid = siddf.std(axis=1)
        ylocs = sep * np.arange(ndemos) + barwidth * i * 2
        plt.barh(ylocs, mean_sid, 2 * barwidth, align='center', label=labels[i], xerr=std_sid,
                 alpha=0.7, color='C%d'%(i-1))

        plt.yticks(sep * np.arange(ndemos) + barwidth * float(len(models) + 1) / 2., demos_)

        plt.xlabel(r'RMS Error [$ms^{-2}$]')

        plt.legend()

        plt.tight_layout()

        plt.savefig(f'{name}.pdf')

        plt.show()

    if figsize == "large":
        fig = plt.figure(figsize=(6, 7))
        plot(fig, f'{savename}_large', large=True)
    else:
        fig = plt.figure()
        plot(fig, savename)


if __name__ == '__main__':
    run()
