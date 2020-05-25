#
#  File: prepare_residual_dataset.py
#
# Prepare the residual from Naive Model
#

import os

import click
import numpy as np
import torch

from ceem.baseline_utils import *
from ceem.data_utils import load_helidata

from scipy.io import savemat

opj = os.path.join


@click.command()
@click.option('--datadir', type=str, default='./datasets/split_normalized')
@click.option('--naivepath', type=str, default='./pretrained_models/naive_baseline.th')
def main(datadir, naivepath):

    torch.set_default_dtype(torch.float64)

    # load naive model
    H = 1
    neural_net_kwargs = DEFAULT_NN_KWARGS(H)

    pfm = LagModel(neural_net_kwargs, H)

    pfm.load_state_dict(torch.load(naivepath))

    # load train data
    train_inputs, train_tgts = load_helidata(datadir, 'train')

    preds = pfm(train_inputs.unsqueeze(2).view(-1, 1, 10)).view(*train_tgts.shape)

    resids = train_tgts - preds

    train_inputs = train_inputs.detach().numpy()
    resids = resids.detach().numpy()

    np.savez(opj(datadir, 'train_resid_data.npz'), u=train_inputs, y=resids)

    savemat(os.path.join(datadir,'n4siddata.mat'), {'inputs':train_inputs, 'targets': resids})


    # load test data
    test_inputs, test_tgts = load_helidata(datadir, 'test')

    preds = pfm(test_inputs.unsqueeze(2).view(-1, 1, 10)).view(*test_tgts.shape)

    resids = test_tgts - preds

    test_inputs = test_inputs.detach().numpy()
    resids = resids.detach().numpy()

    np.savez(opj(datadir, 'test_resid_data.npz'), u=test_inputs, y=resids)

    # load validation data
    valid_inputs, valid_tgts = load_helidata(datadir, 'valid')

    preds = pfm(valid_inputs.unsqueeze(2).view(-1, 1, 10)).view(*valid_tgts.shape)

    resids = valid_tgts - preds

    valid_inputs = valid_inputs.detach().numpy()
    resids = resids.detach().numpy()

    np.savez(opj(datadir, 'valid_resid_data.npz'), u=valid_inputs, y=resids)

    print('Done.')


if __name__ == '__main__':
    main()
