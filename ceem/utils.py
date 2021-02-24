# File: utils.py
#

import random
import resource
import sys
import timeit
from contextlib import contextmanager

import numpy as np
import torch


class Timer(object):

    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


class objectview(object):

    def __init__(self, d):
        self.__dict__ = d

    def __repr__(self):
        return str(self.__dict__)


def peak_memory_mb() -> float:
    """
    Get peak memory usage for this process, as measured by
    max-resident-set size:
    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size
    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in ('linux', 'darwin'):
        return 0.0

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == 'darwin':
        # On OSX the result is in bytes.
        return peak / 1_000_000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1_000


def set_rng_seed(rng_seed: int) -> None:
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)

def require_and_zero_grads(vs):
    for v in vs:
        v.requires_grad_(True)
        try:
            v.grad.zero_()
        except AttributeError:
            pass

def disable_grad(vs):
    for v in vs:
        v.detach_()


@contextmanager
def temp_require_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    require_and_zero_grads(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


@contextmanager
def temp_disable_grad(vs):
    prev_grad_status = [v.requires_grad for v in vs]
    disable_grad(vs)
    yield
    for v, status in zip(vs, prev_grad_status):
        v.requires_grad_(status)


def get_grad_norm(params):
    # check grad norm
    total_norm = 0.
    for p in params:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**(1. / 2)
    return total_norm
