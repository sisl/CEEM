#
# File: learner.py
#

from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from torch.autograd import backward
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ceem import utils
from ceem.opt_criteria import GroupCriterion, STRParamCriterion


def learner(criterion_list, criterion_x_list, opt_list, params_list, crit_kwargs_list,
            opt_kwargs_list=None, subsetinds=None):
    """
    Generic learner function
    Args:
        criterion_list: list of Criterion instances
        criterion_x_list: list of torch.tensor to pass to criteria
        opt_list: list of optimizers (see OPTIMIZERS)
        params_list: list of list of parameters for each optimizer
        crit_kwargs_list: list of kwargs for each criterion
        opt_kwargs_list: kwargs to optimizer
        subsetinds: array specifying the batch indices to operate on

    Returns:
      opt_result_list (list):
    """
    if opt_kwargs_list is None:
        opt_kwargs_list = [DEFAULT_OPTIMIZER_KWARGS[opt] for opt in opt_list]

    # set all requires_grad to false
    for x, c in zip(criterion_x_list, criterion_list):
        if isinstance(x, list):
            assert subsetinds is None, 'subsetinds not supported for listed x.'
            for x_ in x:
                x_.requires_grad_(False)
            if isinstance(c, list):
                assert len(x) == len(c), 'Mismatch between len(criteria), len(xs).'
        else:
            x.requires_grad_(False)

    opt_result_list = []
    for i in range(len(criterion_list)):
        criterion = criterion_list[i]
        criterion_x = criterion_x_list[i]
        opt = OPTIMIZERS[opt_list[i]]
        params = params_list[i]
        crit_kwargs = crit_kwargs_list[i]
        crit_kwargs['inds'] = subsetinds
        opt_kwargs = opt_kwargs_list[i]

        opt_result = opt(criterion, criterion_x, params, crit_kwargs, opt_kwargs)

        opt_result_list.append(opt_result)

    return opt_result_list


def scipy_minimize(criterion, criterion_x, params, crit_kwargs, opt_kwargs):
    """ Wrapper function to call scipy optimizers
    """

    opt_kwargs = deepcopy(opt_kwargs)

    if 'tr_rho' in opt_kwargs:
        tr_rho = opt_kwargs.pop('tr_rho')
        if isinstance(criterion, list):
            criteria = criterion 
            criterion = []
            for c in criteria:
                criterion.append(GroupCriterion([c, STRParamCriterion(tr_rho, params)]))
        else:
            criterion = GroupCriterion([criterion, STRParamCriterion(tr_rho, params)])

    # B, T, n = criterion_x.shape

    vparams0 = parameters_to_vector(params).clone().detach()

    def eval_f(vparams):
        vparams = torch.tensor(vparams).to(torch.get_default_dtype())
        vector_to_parameters(vparams, params)

        with torch.no_grad():
            if isinstance(criterion_x, list):
                loss = 0.
                if isinstance(criterion, list):
                    for (x_,c_) in zip(criterion_x, criterion):
                        loss = loss + c_(x_,**crit_kwargs)
                else:
                    for x_ in criterion_x:
                        loss = loss + criterion(x_, **crit_kwargs)
            else:
                loss = criterion(criterion_x, **crit_kwargs)

        vector_to_parameters(vparams0, params)

        return loss.detach().numpy()

    def eval_g(vparams):
        vparams = torch.tensor(vparams).to(torch.get_default_dtype())
        vector_to_parameters(vparams, params)

        if isinstance(criterion_x, list):
            loss = 0.
            if isinstance(criterion, list):
                for (x_,c_) in zip(criterion_x, criterion):
                    loss = loss + c_(x_,**crit_kwargs)
            else:
                for x_ in criterion_x:
                    loss = loss + criterion(x_, **crit_kwargs)
        else:
            loss = criterion(criterion_x, **crit_kwargs)

        loss.backward()

        grads = torch.cat([
            p.grad.view(-1) if p.grad is not None else torch.zeros_like(p).view(-1) for p in params
        ])

        grads = grads.detach().numpy()

        vector_to_parameters(vparams0, params)

        return grads

    start_feval = eval_f(vparams0.numpy())
    start_gradnorm = np.linalg.norm(eval_g(vparams0.numpy()))

    with utils.Timer() as time:
        opt_result_ = minimize(eval_f, vparams0.numpy(), jac=eval_g, **opt_kwargs)

    vparams = opt_result_['x']

    end_feval = eval_f(vparams)
    end_gradnorm = np.linalg.norm(eval_g(vparams))

    vparams = torch.tensor(vparams).to(torch.get_default_dtype())
    vector_to_parameters(vparams, params)

    net_update_norm = (vparams - vparams0).norm()

    opt_result = {'net_update_norm': net_update_norm.detach().item()}
    opt_result['start_feval'] = float(start_feval)
    opt_result['start_gradnorm'] = start_gradnorm
    opt_result['end_feval'] = float(end_feval)
    opt_result['end_gradnorm'] = end_gradnorm
    opt_result['time'] = time.dt

    return opt_result


TORCH_OPTIMIZERS = {'LBFGS': torch.optim.LBFGS, 'SGD': torch.optim.SGD, 'Adam': torch.optim.Adam}


def ensure_default_torch_kwargs(opt_kwargs):

    if 'method' not in opt_kwargs:
        opt_kwargs['method'] = 'LBFGS'
    if 'lr' not in opt_kwargs:
        if opt_kwargs['method'] == 'LBFGS':
            opt_kwargs['lr'] = 1e-1
        else:
            opt_kwargs['lr'] = 1e-4
    if 'nepochs' not in opt_kwargs:
        opt_kwargs['nepochs'] = 100
    if 'max_grad_norm' not in opt_kwargs:
        opt_kwargs['max_grad_norm'] = 1.0

    return opt_kwargs


def torch_minimize(criterion, criterion_x, params, crit_kwargs, opt_kwargs):
    """ Wrapper function to use torch.optim optimizers
    """
    opt_kwargs = deepcopy(opt_kwargs)

    opt_kwargs = ensure_default_torch_kwargs(opt_kwargs)

    if 'tr_rho' in opt_kwargs:
        tr_rho = opt_kwargs.pop('tr_rho')
        if isinstance(criterion, list):
            for i, c in enumerate(criterion):
                criterion[i] = GroupCriterion([c, STRParamCriterion(tr_rho, params)])
        else:
            criterion = GroupCriterion([criterion, STRParamCriterion(tr_rho, params)])

    method = opt_kwargs.pop('method')

    nepochs = opt_kwargs.pop('nepochs')

    max_grad_norm = opt_kwargs.pop('max_grad_norm')

    opt = TORCH_OPTIMIZERS[method](params, **opt_kwargs)

    def closure():
        opt.zero_grad()
        if isinstance(criterion_x, list):
            loss = 0.
            if isinstance(criterion, list):
                for (x_,c_) in zip(criterion_x, criterion):
                    loss = loss + c_(x_,**crit_kwargs)
            else:
                for x_ in criterion_x:
                    loss = loss + criterion(x_, **crit_kwargs)
        else:
            loss = criterion(criterion_x, **crit_kwargs)

        loss.backward()
        return loss

    start_feval = closure()
    start_gradnorm = utils.get_grad_norm(params)

    vparams0 = parameters_to_vector(params).clone().detach()

    with utils.Timer() as time:
        for epoch in range(nepochs):

            if method == 'LBFGS':
                opt.step(closure=closure)
                loss = closure()
            else:
                loss = closure()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()

            if epoch % 100 == 0:
                print('Epoch %d, Loss %.3e' % (epoch, float(loss)))

    end_feval = closure()
    end_gradnorm = utils.get_grad_norm(params)

    vparams = parameters_to_vector(params).clone().detach()

    net_update_norm = (vparams - vparams0).norm()

    opt_result = {
        'time': time.dt,
        'start_feval': start_feval.detach().item(),
        'start_gradnorm': start_gradnorm,
        'end_feval': end_feval.detach().item(),
        'end_gradnorm': end_gradnorm,
        'net_update_norm': net_update_norm.detach().item()
    }
    return opt_result


OPTIMIZERS = {'scipy_minimize': scipy_minimize, 'torch_minimize': torch_minimize}

DEFAULT_OPTIMIZER_KWARGS = {
    'scipy_minimize': {
        'method': 'Nelder-Mead'
    },
    'torch_minimize': ensure_default_torch_kwargs({})
}
