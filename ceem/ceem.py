#
# File: ceem.py
#
from typing import Any, Callable, Tuple

import joblib
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ceem import logger, nested
from ceem.learner import DEFAULT_OPTIMIZER_KWARGS, learner
from ceem.smoother import NLSsmoother
from tqdm import tqdm




class CEEM:

    def __init__(self, smoothing_criteria: Tuple, learning_criteria: Tuple, learning_params: Tuple,
                 learning_opts: Tuple, epoch_callbacks: Tuple[Callable[[int], None], ...],
                 termination_callback: Callable[[int], bool], parallel: int = 0):
        """
        Args:
          smoothing_criteria (list): len B list of smoothing criterion
          learning_criteria (list): len >=1 list of learning criterion
          learning_params (list): len(learning_criteria) list of param groups
          learning_opts (list): len(learning_criteria) list of learning optimizers
          epoch_callbacks (list): functions accepting epoch as argument
          termination_callback (callable): function accepting epoch as argument, returning True/False
          parallel (int): Number of processes to parallelize smoothing batch of trajectories
        """
        self._smoothing_criteria = smoothing_criteria
        self._learning_criteria = learning_criteria
        self._learning_params = learning_params
        self._learning_opts = learning_opts
        self._epoch_callbacks = epoch_callbacks
        self._termination_callback = termination_callback
        self._parallel = parallel

    def smooth(self, xsm, sys, solver_kwargs=None, subsetinds=None):
        """
        Args:
          xsm (torch.tensor): (B,T,n) trajectories of states
          sys (DiscreteDynamicalSystem):

        Returns:
          xsm (torch.tensor): (B,T,n) trajectories of smoothed states
          metrics_l (list): 
        """
        if solver_kwargs is None:
            solver_kwargs = {'verbose': 0}

        B = len(self._smoothing_criteria)
        xsm_l = []
        metrics_l = []
        iterator = list(range(B)) if subsetinds is None else subsetinds

        # Run smoothers
        if self._parallel > 0:
            results = joblib.Parallel(n_jobs=self._parallel, backend="loky")(
                joblib.delayed(NLSsmoother)(x0=xsm[b:b + 1], criterion=self._smoothing_criteria[b],
                                            system=sys, solver_kwargs=solver_kwargs)
                for b in tqdm(iterator))
            xsm_l, metrics_l = nested.zip(*results)
        else:
            for b in tqdm(iterator):
                xsm_, metrics = NLSsmoother(xsm[b:b + 1], self._smoothing_criteria[b], sys,
                                            solver_kwargs=solver_kwargs)
                xsm_l.append(xsm_)
                metrics_l.append(metrics)

        for i, b in enumerate(iterator):
            xsm[b] = xsm_l[i]

        return xsm, metrics_l

    def step(self, xs, sys, smooth_solver_kwargs=None, learner_criterion_kwargs=None,
             learner_opt_kwargs=None, subset=None):
        """ Runs one step of CEEM algorithm

        Args:
          xs (torch.tensor):
          sys (DiscreteDynamicalSystem):
          smooth_solver_kwargs (dict):
          learner_criterion_kwargs (dict):
          learner_opt_kwargs (dict):
          subset (int):

        Returns:
          xs (torch.tensor):
          smooth_metrics (dict):
          learn_metrics (dict):

        First runs the smoothing step on trajectories `xs`.
        Then runs the learning step on `sys`.
        """

        if subset is not None:
            B = xs.shape[0]
            subB = min(subset, B)
            subsetinds = np.random.choice(B, subB, replace=False)
        else:
            subsetinds = None

        if learner_criterion_kwargs is None:
            learner_criterion_kwargs = {}

        if isinstance(learner_criterion_kwargs, dict):
            learner_criterion_kwargs = [
                learner_criterion_kwargs for _ in range(len(self._learning_criteria))
            ]

        if learner_opt_kwargs is None:
            learner_opt_kwargs = [DEFAULT_OPTIMIZER_KWARGS[opt] for opt in self._learning_opts]

        if isinstance(learner_opt_kwargs, dict):
            learner_opt_kwargs = [learner_opt_kwargs for _ in range(len(self._learning_criteria))]

        assert len(learner_criterion_kwargs) == len(self._learning_criteria)
        assert len(learner_opt_kwargs) == len(self._learning_criteria)

        logger.info('Executing smoothing step')
        xs, smooth_metrics = self.smooth(xs, sys, smooth_solver_kwargs, subsetinds=subsetinds)

        logger.info('Executing learning step')
        ncrit = len(self._learning_criteria)

        # Learn theta
        learn_metrics = learner(sys, self._learning_criteria, [xs] * ncrit, self._learning_opts,
                                self._learning_params, learner_criterion_kwargs,
                                opt_kwargs_list=learner_opt_kwargs, subsetinds=subsetinds)

        if isinstance(smooth_metrics, (list, tuple)):
            smooth_metrics = nested.zip(*smooth_metrics)

        if isinstance(learn_metrics, (list, tuple)):
            learn_metrics = nested.zip(*learn_metrics)

        return xs, smooth_metrics, learn_metrics

    def train(self, xs, sys, nepochs, smooth_solver_kwargs=None, learner_criterion_kwargs=None,
              learner_opt_kwargs=None, subset=None):
        """ Runs one step of CEEM algorithm

        Args:
          xs (torch.tensor):
          sys (DiscreteDynamicalSystem):
          nepochs (int): Number of epochs to run CEEM algorithm for
          smooth_solver_kwargs (dict):
          learner_criterion_kwargs (dict):
          learner_opt_kwargs (dict):
          subset (int):
        """
        for epoch in range(nepochs):
            xs, smooth_metrics, learn_metrics = self.step(xs, sys, smooth_solver_kwargs,
                                                          learner_criterion_kwargs,
                                                          learner_opt_kwargs, subset=subset)

            # Log all metrics. Run tensorboard to visualize
            log_kv_or_listkv(smooth_metrics, "smooth")
            log_kv_or_listkv(learn_metrics, "learn")

            for ecall in self._epoch_callbacks:
                ecall(epoch)

            if self._termination_callback(epoch):
                break

            logger.dumpkvs()


def log_kv_or_listkv(dic, prefix):
    for k, v in dic.items():
        if isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                if isinstance(val, (float, int)):
                    logger.logkv("{}/{}/{}".format(prefix, k, i), val)
        elif isinstance(v, (float, int)):
            logger.logkv("{}/{}".format(prefix, k), v)
