import os
import torch

opj = os.path.join

device = 'cpu'


def load_helidata(datadir, split, return_files=False):
    data = []
    controls = []
    files = os.listdir(opj(datadir, f'{split}'))
    for f in files:
        if f[0] == '.':
            continue
        time, dat, cont = torch.load(opj(f'{datadir}', f'{split}', f'{f}'))
        controls.append(cont)
        data.append(dat)

    data = torch.stack(data, dim=0)
    cont = torch.stack(controls, dim=0)
    target = data[:, :, -6:]
    data = torch.cat((cont, data[:, :, 7:-6]), dim=2)
    data = data.to(device)
    target = target.to(device)
    if return_files:
        return data, target, files
    else:
        return data, target


def load_statistics(datadir):
    data_mean, data_std, controls_mean, controls_std = torch.load(opj(datadir, 'statistics.pt'))
    y_mean = data_mean[-6:]
    u_mean = torch.cat([controls_mean, data_mean[7:-6]], dim=-1)
    y_std = data_std[-6:]
    u_std = torch.cat([controls_std, data_std[7:-6]], dim=-1)
    return y_mean, y_std, u_mean, u_std
