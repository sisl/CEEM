#
# File: odesolver.py
#

import abc
import torch


class Euler:

    @staticmethod
    def step_func(func, t, dt, y, u, transforms=None):
        return tuple(dt * f_ for f_ in func(t, y, u=u))

    @property
    def order(self):
        return 1


class Midpoint:

    @staticmethod
    def step_func(func, t, dt, y, u, transforms=None):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t, y, u=u)))
        y_mid = tuple(trans(y_) for y_, trans in zip(y_mid, transforms))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid, u=u))

    @property
    def order(self):
        return 2


class RK4:

    @staticmethod
    def step_func(func, t, dt, y, u, transforms=None):
        return rk4_alt_step_func(func, t, dt, y, u=u)

    @property
    def order(self):
        return 4


def rk4_alt_step_func(func, t, dt, y, k1=None, u=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y, u=u)
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)), u=u)
    k3 = func(t + dt * 2 / 3,
              tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_, k2_ in zip(y, k1, k2)), u=u)
    k4 = func(t + dt,
              tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)), u=u)
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8)
                 for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


def odestep(func, t, dt, y0, u=None, method='midpoint', transforms=None):
    tensor_input, func, y0, t = _check_inputs(func, y0, t)
    if transforms is None:
        transforms = [lambda x: x for _ in range(len(y0))]

    dy = SOLVERS[method].step_func(func, t, dt, y0, u=u, transforms=transforms)
    y = tuple(trans(y0_ + dy_) for y0_, dy_, trans in zip(y0, dy, transforms))
    if tensor_input:
        y = y[0]

    return y


SOLVERS = {
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
}


def _check_inputs(func, y0, t):

    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)

        _base_nontuple_func_ = func
        func = lambda t, y, u: (_base_nontuple_func_(t, y[0], u),)
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'

    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(
            type(y0_))

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))

    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t
