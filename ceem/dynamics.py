#
# File: dynamics.py
#
import abc
from collections import OrderedDict

import torch

from .odesolver import odestep

# DynamicalSystems


class DiscreteDynamicalSystem(metaclass=abc.ABCMeta):
    """

    """

    def step(self, t, x, u=None):
        """Returns next x_{t+1}

        Args:
            t (torch.IntTensor): (B, T,) shaped time
            x (torch.tensor): (B, T, n) shaped system states
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            x_next (torch.tensor): (B, T, n) shaped next system states
        """
        raise NotImplementedError

    def observe(self, t, x, u=None):
        """Returns y_t"""
        raise NotImplementedError

    @property
    def xdim(self):
        """system state dimensions
        """
        return self._xdim

    @property
    def udim(self):
        """system input dimension
        returns None if unactuated
        """
        return self._udim

    @property
    def ydim(self):
        """system observation dimension
        """
        return self._ydim


class ContinuousDynamicalSystem:
    """

    """

    def step_derivs(self, t, x, u=None):
        """Returns xdot_t

        Args:
            t (torch.tensor): (B, T,) shaped time indices
            x (torch.tensor): (B, T, n) shaped system states
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            xdot (torch.tensor): (B, T, n) system state derivatives
        """
        raise NotImplementedError

    def observe(self, t, x, u=None):
        """Returns y_t"""
        raise NotImplementedError

    @property
    def xdim(self):
        """system state dimensions
        """
        return self._xdim

    @property
    def udim(self):
        """system input dimension
        returns None if unactuated
        """
        return self._udim

    @property
    def ydim(self):
        """system observation dimension
        """
        return self._ydim


class KinoDynamicalSystem(ContinuousDynamicalSystem):
    """

    """

    def step_derivs(self, t, x, u=None):
        """Returns xdot_t

        Args:
            t (torch.tensor): (B, T,) shaped time indices
            x (torch.tensor): (B, T, qn+vn) shaped system states
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            xdot (torch.tensor): (B, T, n) system state derivatives
        """
        q = x[:, :, :self.qdim]
        v = x[:, :, self.qdim:]

        qdot = self.kinematics(t, q, v, u)
        vdot = self.dynamics(t, q, v, u)

        xdot = torch.cat([qdot, vdot], dim=-1)

        return xdot

    def kinematics(self, t, q, v, u):
        """Returns qdot_t

        Args:
            t (torch.tensor): (B, T,) shaped time indices
            q (torch.tensor): (B, T, qn) shaped system generalized coordinates
            v (torch.tensor): (B, T, vn) shaped system generalized velocities
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            qdot (torch.tensor): (B, T, n) system gen. coordinate derivatives
        """
        raise NotImplementedError

    def dynamics(self, t, q, v, u):
        """Returns qdot_t

        Args:
            t (torch.tensor): (B, T,) shaped time indices
            q (torch.tensor): (B, T, qn) shaped system generalized coordinates
            v (torch.tensor): (B, T, vn) shaped system generalized velocities
            u (torch.tensor): (B, T, m) shaped control inputs

        Returns:
            vdot (torch.tensor): (B, T, n) system gen. velocity derivatives
        """
        raise NotImplementedError

    @property
    def qdim(self):
        return self._qdim

    @property
    def vdim(self):
        return self._vdim

    @property
    def xdim(self):
        return self.qdim + self.vdim


class C2DSystem(DiscreteDynamicalSystem, ContinuousDynamicalSystem):
    """
    Continuous to Discrete System

    """

    def __init__(self, dt, method, transforms=None):
        """
        Args:
            csys (ContinuousTimeSystem): continuous-time system
            dt (float): time-step
            method (str): see ceem/odesolver.py SOLVERS for options
            transforms (tuple)
        """
        self._dt = dt
        self._method = method
        self._transforms = transforms

    def step(self, t, x, u=None):
        nx = odestep(self.step_derivs, t, self._dt, x, u=u, method=self._method,
                     transforms=self._transforms)
        return nx


class ObsJacMixin:
    """
    Mixin for computing jacobians of observations.
    Defaults to using autograd.
    """

    def jac_obs_x(self, t, x, u=None):
        """Returns the Jacobian of observation wrt x

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_y_t (torch.tensor): (B, T, ydim, n) shaped jacobian of the observation
        """

        p = self._ydim

        x = x.detach()
        x.requires_grad_(True)

        y = self.observe(t, x, u)

        jac_x = torch.stack(
            [torch.autograd.grad(y[:, :, i].sum(), x, retain_graph=True)[0] for i in range(p)],
            dim=2)

        return jac_x

    def jac_obs_u(self, t, x, u):
        """Returns the Jacobian of observation wrt u

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_y_t (torch.tensor): (B, T, ydim, m) shaped jacobian of the observation
        """

        p = self._ydim

        u = u.detach()
        u.requires_grad_(True)

        y = self.observe(t, x, u)

        jac_u = torch.stack(
            [torch.autograd.grad(y[:, :, i].sum(), u, retain_graph=True)[0] for i in range(p)],
            dim=2)

        return jac_u

    def jac_obs_theta(self, t, x, u=None):
        """Returns the Jacobian of observation wrt theta

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_y_t (seq of torch.tensor): jacobian of the observation wrt each nn.Parameter
        """
        # TODO - hopefully we bypass ever explicitly calling this
        raise NotImplementedError


class DynJacMixin:
    """
    Mixin for computing jacobians of dynamics.
    Defaults to using autograd.
    """

    def jac_step_x(self, t, x, u=None):
        """Returns the Jacobian of step at time t

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_x_t (torch.tensor): (B, T, n, n) shaped jacobian of the next state
        """
        n = self.xdim

        x = x.detach()
        x.requires_grad_(True)

        nx = self.step(t, x, u)

        jac_x = torch.stack(
            [torch.autograd.grad(nx[:, :, i].sum(), x, retain_graph=True)[0] for i in range(n)],
            dim=2)

        return jac_x

    def jac_step_u(self, t, x, u):
        """Returns the Jacobian of step at time t

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_u (torch.tensor): (B, T, n, m) shaped jacobian of the next state
        """
        n = self.xdim

        u = u.detach()
        u.requires_grad_(True)

        nx = self.step(t, x, u)

        jac_u = torch.stack(
            [torch.autograd.grad(nx[:, :, i].sum(), u, retain_graph=True)[0] for i in range(n)],
            dim=2)

        return jac_u

    def jac_step_theta(self, t, x, u=None):
        """Returns the Jacobian of step at time t

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_x_t (seq of torch.tensor): jacobian of the observation wrt each nn.Parameter
        """
        nx = self.step(t, x, u)
        B, T, N = x.shape

        jacobians = OrderedDict([(name, None) for name, p in self.named_parameters()])

        for iB in range(B):
            for iT in range(T):
                for iN in range(N):
                    self.zero_grad()
                    nx[iB, iT, iN].backward(retain_graph=True)
                    for name, param in self.named_parameters():
                        if jacobians[name] is None:
                            jacobians[name] = torch.zeros(B, T, N, *param.shape)
                        if param.grad is not None:
                            jacobians[name][iB, iT, iN] = param.grad
        return jacobians


class AnalyticObsJacMixin(ObsJacMixin):
    """
    Mixin for analytically computing observation jacobains
    """

    def jac_obs_x(self, t, x, u=None):
        """Returns the Jacobian of observation wrt x

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_y_t (torch.tensor): (B, T, m, n) shaped jacobian of the observation
        """

        raise NotImplementedError

    def jac_obs_theta(self, t, x, u=None):
        """Returns the Jacobian of observation wrt theta

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_y_t (seq of torch.tensor): jacobian of the observation wrt each nn.Parameter
        """
        raise NotImplementedError


class AnalyticDynJacMixin(DynJacMixin):
    """
    Mixin for analytically computing jacobians of dynamics.
    """

    def jac_step_x(self, t, x, u=None):
        """Returns the Jacobian of step at time t

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_x_t (torch.tensor): (B, T, n, n) shaped jacobian of the next state
        """
        raise NotImplementedError

    def jac_step_theta(self, t, x, u=None):
        """Returns the Jacobian of step at time t

        Args:
           t (torch.tensor): (B, T,) shaped time indices
           x (torch.tensor): (B, T, n) shaped system states
           u (torch.tensor): (B, T, m) shaped control inputs

        Returns
           jac_x_t (seq of torch.tensor): jacobian of the observation wrt each nn.Parameter
        """
        # TODO - hopefully we bypass ever explicitly calling this
        raise NotImplementedError
