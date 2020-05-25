import torch
import numpy as np
from ceem.baseline_utils import LagModel
from ceem.systems import DiscreteLinear
from ceem.data_utils import *
from ceem.smoother import EKF

from scipy.io import loadmat

def compute_rms(y, ypred, y_std=None):
    """
    Compute RMS prediction error on type of trajectory
    Args: y (torch.tensor): (B,T,m) tensor of true observations
          ypred (torch.tensor): (B,T,m) tensor of predicted observations
          demonames (list): len=T list of demo names
          y_std (torch.tensor): (m,) standard deviations
    Returns:
          (B,) rms errors on each demo
    """
    B,T,m = y.shape
    if y_std is None:
        y_std = torch.ones(y.shape[-1])

    all_rms = ((y-ypred)* y_std.view(1,1,m))**2 
    all_rms = all_rms.sum(-1).sum(-1) / (T*m)
    all_rms = all_rms.sqrt()
    return all_rms

def gen_ypred_benchmark(net, H, u):
    
    B,T,m = u.shape
    ypreds = []
    for t in range(T-H+1):
        D = u[:, t:t+H]
        ypreds.append(net(D))
    ypred = torch.stack(ypreds,dim=1)

    return ypred

def gen_ypred_model(model, u, y, s0fac=1.0, rfac=1.0, qfac=1.0):
    xdim = model._xdim
    B,T,udim = u.shape
    _,_,ydim = y.shape

    x0 = torch.zeros(B,1,xdim)
    sigma0 = torch.eye(xdim) * s0fac
    Q = torch.eye(xdim) * qfac
    R = torch.eye(ydim) * rfac

    x, ypred = EKF(x0, y, u, sigma0, Q, R, model)

    return ypred