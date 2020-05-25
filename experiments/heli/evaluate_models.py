import os

import numpy as np
import torch
from scipy.io import loadmat

import pandas as pd
from ceem.baseline_utils import LagModel
from ceem.data_utils import *
from ceem.exp_utils import *
from ceem.smoother import EKF
from ceem.systems import DiscreteLinear

import click


@click.command()
@click.option('--datadir', default='./datasets/split_normalized', type=click.Path())
@click.option('--naivepath', default='./pretrained_models/naive_baseline.th', type=click.Path())
@click.option('--h25path', default='./pretrained_models/h25_baseline.th', type=click.Path())
@click.option('--nlpath', default='./pretrained_models/NL_model.th', type=click.Path())
@click.option('--sidpath', default='./pretrained_models/SID', type=click.Path())
@click.option('--lstmpath', default='./pretrained_models/lstm.th', type=click.Path())
@click.option('--savedir', default='./experiments/heli/trajectories', type=click.Path())
def main(datadir, naivepath, h25path, nlpath, sidpath, lstmpath, savedir):
    torch.set_default_dtype(torch.float64)

    # load test data
    test_u, test_y, demos = load_helidata(datadir, 'test', return_files=True)
    y_mean, y_std, _, _ = load_statistics(datadir)
    xdim = 10
    _, _, udim = test_u.shape
    _, _, ydim = test_y.shape

    # load baseline models
    naivemodel, h25model = load_baseline(naivepath, h25path)

    # load lstm baseline model
    lstmmodel = load_lstm_baseline(lstmpath)

    # load SID models
    sid10Dmodels = []
    for f in os.listdir(sidpath):
        if 'SID_10D' in f:
            sid10Dmodels.append(load_sid_model(opj(sidpath, f), xdim, udim, ydim))

    # load NL model
    nlmodel = DiscreteLinear(xdim, udim, ydim)
    nlmodel.load_state_dict(torch.load(nlpath))

    # evaluate baseline models
    np_ = gen_ypred_benchmark(naivemodel, 1, test_u)
    naivepred = np_ * y_std + y_mean
    h25pred = gen_ypred_benchmark(h25model, 25, test_u) * y_std + y_mean

    lstmpred = gen_ypred_benchmark_lstm(lstmmodel, 25, test_u) * y_std + y_mean

    sid10D_preds = []
    for sidmodel in sid10Dmodels:
        sid10D_preds.append((gen_ypred_model(sidmodel, test_u, test_y - np_) + np_) * y_std +
                            y_mean)

    # evaluate nl model
    nlpred = (gen_ypred_model(nlmodel, test_u, test_y - np_) + np_) * y_std + y_mean

    # compute rms errors
    naive_rms = compute_rms(test_y[:, 25:] * y_std + y_mean, naivepred[:, 25:])
    h25_rms = compute_rms(test_y[:, 25:] * y_std + y_mean, h25pred[:, 1:])
    nl_rms = compute_rms(test_y[:, 25:] * y_std + y_mean, nlpred[:, 25:])
    lstm_rms = compute_rms(test_y[:, 25:] * y_std + y_mean, lstmpred[:, 25:])

    print('Naive mean rmse: %.3f' % naive_rms.mean())
    print('H25 mean rmse: %.3f' % h25_rms.mean())
    print('LSTM mean rmse: %.3f' % lstm_rms.mean())
    print('dl10d_mean_rmse: %.3f' % nl_rms.mean())
    print('Naive is %.3f times worse than H25', naive_rms.mean() / h25_rms.mean())
    print('LSTM is %.3f times worse than H25', lstm_rms.mean() / h25_rms.mean())


    df = pd.DataFrame(data=dict(
        demos=demos,
        naive=naive_rms.detach().numpy(),
        H25=h25_rms.detach().numpy(),
        lstm=lstm_rms.detach().numpy(),
        #SID=sid_rms.detach().numpy(),
        NL=nl_rms.detach().numpy()))

    for i, sid10D_pred in enumerate(sid10D_preds):
        sid10D_rms = compute_rms(test_y[:, 25:] * y_std + y_mean, sid10D_pred[:, 25:])

        print('sid10d_%d_mean_rmse: %.3f' % (i, sid10D_rms.mean()))

        df['SID10D_%d' % i] = sid10D_rms.detach().numpy()

    # Save data
    os.makedirs(savedir, exist_ok=True)
    df.to_pickle(f'{savedir}/evaluations.pkl')
    torch.save(naivepred.detach(), f'{savedir}/naivepred')
    torch.save(h25pred.detach(), f'{savedir}/h25pred')
    torch.save(sid10D_preds[0].detach(), f'{savedir}/sidpred')
    torch.save(nlpred.detach(), f'{savedir}/nlpred')
    torch.save(lstmpred.detach(), f'{savedir}/lstmpred')


def load_sid_model(matpath, xdim, udim, ydim):
    tt = lambda x: torch.tensor(x).to(torch.get_default_dtype())
    params = loadmat(matpath)
    xdim = params['A'].shape[0]
    model = DiscreteLinear(xdim, udim, ydim, A=tt(params['A']), B=tt(params['B']),
                           C=tt(params['C']), D=tt(params['D']), obsModel=None)
    return model


def load_baseline(naivepath, h25path):

    H = 1
    naive_neural_net_kwargs = dict(input_size=10 * H, hidden_sizes=[32] * 8, output_size=6,
                                   activation='tanh', gain=1.0, ln=False)
    naivemodel = LagModel(naive_neural_net_kwargs, H)
    naivemodel.load_state_dict(torch.load(naivepath))

    H = 25
    h25_neural_net_kwargs = dict(input_size=10 * H, hidden_sizes=[32] * 8, output_size=6,
                                 activation='tanh', gain=1.0, ln=False)

    h25model = LagModel(h25_neural_net_kwargs, H)
    h25model.load_state_dict(torch.load(h25path))

    return naivemodel, h25model


class EvalLSTMModel2(torch.nn.Module):

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


def load_lstm_baseline(lstmpath):
    DEFAULT_LSTM_KWARGS = dict(
        input_size=10,
        hidden_size=32,
        num_layers=2,
        output_size=6,)
    lstmmodel = EvalLSTMModel2(**DEFAULT_LSTM_KWARGS)
    lstmmodel.load_state_dict(torch.load(lstmpath))
    return lstmmodel


def gen_ypred_benchmark_lstm(net, H, u):

    B, T, m = u.shape
    ypred = net(u)
    # ypreds = []
    # for t in range(T - H + 1):
    #     D = u[:, t:t + H]
    #     ypreds.append(net(D))
    #     ypred = torch.stack(ypreds, dim=1)

    return ypred


if __name__ == '__main__':
    main()
