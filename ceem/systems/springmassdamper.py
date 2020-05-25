from collections import OrderedDict

import torch
from torch import nn

from ceem.dynamics import (AnalyticObsJacMixin, C2DSystem, DynJacMixin, KinoDynamicalSystem)


class SpringMassDamper(KinoDynamicalSystem, C2DSystem, nn.Module, DynJacMixin, AnalyticObsJacMixin):
    """Generalized Spring Mass Damper System
    """

    def __init__(self, M, D, K, dt, method='midpoint'):
        """
           Args:
              M (torch.tensor): (qn,qn) positive-definite mass matrix
              D (torch.tensor): (qn,qn) positive-definite damping matrix
              K (torch.tensor): (qn,qn) positive-definite stiffness matrix
        """

        C2DSystem.__init__(self, dt=dt, method=method)
        nn.Module.__init__(self)

        qn = M.shape[0]

        Minv = M.inverse()
        Minv_chol = Minv.unsqueeze(0).cholesky()
        D_chol = D.unsqueeze(0).cholesky()
        K_chol = K.unsqueeze(0).cholesky()

        self._Minv_chol = nn.Parameter(Minv_chol.unsqueeze(0))
        self._D_chol = nn.Parameter(D_chol.unsqueeze(0))
        self._K_chol = nn.Parameter(K_chol.unsqueeze(0))

        self._qdim = qn
        self._vdim = qn

    def dynamics(self, t, q, v, u):

        Minv = self._Minv_chol @ self._Minv_chol.transpose(2, 3)
        D = self._D_chol @ self._D_chol.transpose(2, 3)
        K = self._K_chol @ self._K_chol.transpose(2, 3)

        vdot = -(Minv @ (D @ v.unsqueeze(3) + K @ q.unsqueeze(3))).squeeze(3)

        return vdot

    def kinematics(self, t, q, v, u):
        return v

    def observe(self, t, x, u=None):

        return x * 1.0

    # def kinematics_jacobian(self, t, q, v, u):
    #     B,T,qn = q.shape
    #     dqdotdq = torch.zeros(B,T,self._qdim,self._qdim)
    #     dqdotdv = torch.eye(self._qdim).expand(B,T,self._qdim,self._vdim)

    #     return torch.cat([dqdotdq,dqdotdv], dim=-1)

    # def dynamics_jacobian(self, t, q, v, u):
    #     B,T,qn = q.shape

    #     Minv = self._Minv_chol @ self._Minv_chol.transpose(2, 3)
    #     D = self._D_chol @ self._D_chol.transpose(2, 3)
    #     K = self._K_chol @ self._K_chol.transpose(2, 3)

    #     return -(Minv @ torch.cat([K, D], dim=-1)).expand(B,T,self._vdim,self._qdim + self._vdim)

    # def jac_dyn_x(self, t, x, u):
    #     """
    #     Return jac_x (xdot)
    #     """
    #     B,T,n = x.shape

    #     q = x[:,:,:self._qdim]
    #     v = x[:,:,self._qdim:]

    #     dqdotdx = self.kinematics_jacobian(t,q,v,u)
    #     dvdotdx = self.dynamics_jacobian(t,q,v,u)
    #     return torch.cat([dqdotdx,dvdotdx], dim=-2)

    def jac_obs_x(self, t, x, u):

        B, T, n = x.shape

        return torch.eye(self.xdim).expand(B, T, n, n)

    def jac_obs_theta(self, t, x, u):

        return OrderedDict([('_Minv_chol', None), ('_D_chol', None), ('_K_chol', None)])


def main():
    n = 4

    M = D = K = torch.tensor([[1., 2.], [2., 5.]])

    csys = SpringMassDamper(M, D, K)

    x = torch.randn(2, 2, n)

    # print(csys.jac_dyn_x(None, x, None))


if __name__ == '__main__':
    main()
