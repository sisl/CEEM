import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import click

opj = os.path.join

@click.command()
@click.option('--ceemdatadir', type=str, default='./data/lorenz/convergence_experiment/ceem')
@click.option('--pemdatadir', type=str, default='./data/lorenz/convergence_experiment/pem')
def main(ceemdatadir, pemdatadir):


    fig,ax = plt.subplots(1,figsize=(4.5,3))

    k = 6
    for i, b in enumerate([2,4,8]):
        try:
            for seed in range(42,42+4):
                try:
                    expstr = 'k=%d_B=%d_seed=%d/progress.csv'%(k,b,seed)
                    df = pd.read_csv(opj(ceemdatadir, expstr))

                    error = 10 ** df['test/log10_error'].values
                    print(df['test/log10_error'].values[:10])
                    l,=ax.plot(range(error.shape[0]),
                        error, color='C%d'%i, alpha=0.55)

                    if seed == 42:
                        l.set_label('CE-EM: (%d traj)' % b)

                    if b == 8:

                        expstr = 'k=%d_B=%d_seed=%d/progress.csv'%(k,b,seed)
                        df = pd.read_csv(opj(pemdatadir, expstr))

                        error = 10 ** df['test/log10_error'].values
                        print(df['test/log10_error'].values[:10])
                        l_,=ax.plot(df['time/epoch'],
                            error, color='C%d'%(i+1), alpha=0.55)

                        if seed == 42:
                            l_.set_label('Particle EM (%d traj) ' % b)

                except FileNotFoundError:
                    pass
            
        except UnboundLocalError:
            pass



    ax.set_yscale('log')
    ax.set_xlabel('EM Epoch')
    ax.set_ylabel(r'$\epsilon(\theta)$')
    ax.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(opj('Convergence_k=6.pdf'))
    plt.show()
    
if __name__ == '__main__':
    main()