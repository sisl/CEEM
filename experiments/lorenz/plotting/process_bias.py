import pandas as pd
import os
import click
import numpy as np

opj = os.path.join


@click.command()
@click.option('--datadir', type=str, default='./data/lorenz/bias_experiment')
def main(datadir):

    df = pd.read_pickle(opj(datadir, 'results.pkl'))
    print(df)

    print('\\toprule')
    print('$\\sigma_w$ & $\\sigma_y$  & \\multicolumn{1}{c}{$\\sigma$} & \\multicolumn{1}{c}{$\\rho$} & \\multicolumn{1}{c}{$\\beta$} \\\\ \\midrule')
#  $1\cdot 10^{-2}$&$1\cdot 10^{-2}$        & 10.017 (0.012) & 28.000 (0.001) & 2.668 (0.001) \\
#  $5\cdot 10^{-2}$&$1\cdot 10^{-2}$ & 10.014 (0.018) & 28.002 (0.004) & \outside{2.671 (0.002)} \\ 
#  $1\cdot 10^{-1}$&$1\cdot 10^{-2}$        & 10.051 (0.035) & 27.995 (0.013) & \outside{2.676 (0.003)} \\
#  $1\cdot 10^{-2}$&$5\cdot 10^{-2}$ & 10.015 (0.016) & 27.998 (0.002) & 2.667 (0.001) \\
#  $1\cdot 10^{-2}$&$1\cdot 10^{-1}$        & 10.011 (0.021) & 27.997 (0.004) & 2.666 (0.001) \\ \bottomrule
# \end{tabular}}

    def check(ystd,wstd):
        df_ = df.loc[(df['ystd']==ystd) & (df['wstd']==wstd)]

        means = df_.mean()
        stds = df_.std() / np.sqrt(len(df_))

        sigmam = means['sigma']
        sigmas = stds['sigma']
        rhom = means['rho']
        rhos = stds['rho']
        betam = means['beta']
        betas = stds['beta']

        if np.abs(sigmam - 10.) / sigmas > 2.0:
            sigmastr = '\\outside{%.3f (%.3f)}' % (sigmam, sigmas)
        else:
            sigmastr = '%.3f (%.3f)' % (sigmam,sigmas)

        if np.abs(rhom - 28.) / rhos > 2.0:
            rhostr = '\\outside{%.3f (%.3f)}' % (rhom, rhos)
        else:
            rhostr = '%.3f (%.3f)' % (rhom, rhos)

        if np.abs(betam - 8./3.) / betas > 2.0:
            betastr = '\\outside{%.3f (%.3f)}' % (betam, betas)
        else:
            betastr = '%.3f (%.3f)' % (betam, betas)

        line = '$%.3f$ & $%.2f$ & %s & %s & %s \\\\' % (wstd, ystd, sigmastr, rhostr, betastr)
        print(line)

    ystd = 1e-2
    for wstd in [1e-3, 1e-2, 1e-1]:
        check(ystd, wstd)

    wstd = 1e-3
    for ystd in [5e-2, 1e-1]:
        check(ystd, wstd)

        


if __name__ == '__main__':
    main()