#
# File: dynamics_test.py
#
import torch
from ceem.systems import SpringMassDamper, LorenzSystem
from ceem.dynamics import C2DSystem
import matplotlib.pyplot as plt


def test():

    n = 4

    M = D = K = torch.tensor([[1., 2.], [2., 5.]])

    D = torch.eye(2) * 0.001

    dt = 0.1

    method = 'midpoint'

    sys = SpringMassDamper(M, D, K, dt, method=method)

    # sys = C2DSystem(csys, dt, method)

    B = 2

    T = 300

    xs = [torch.randn(B, 1, n)]

    for t in range(T):

        tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
        xs.append(sys.step(tinp, xs[-1]))

    x = torch.cat(xs, dim=1)

    for b in range(B):
        plt.subplot(B, 1, b + 1)
        plt.plot(range(T + 1), x[b, :, :2].detach().numpy())

    plt.title('SpringMassDamper')

    # plt.show()

    k = 2
    dt = 0.04

    sigma = torch.tensor([10.] * k)
    rho = torch.tensor([28.] * k)
    beta = torch.tensor([8. / 3.] * k)

    H = torch.randn(3 * k, 3 * k)

    C = torch.randn(3 * k - 3, 3 * k)

    n = 3 * k

    method = 'midpoint'

    sys = LorenzSystem(sigma, rho, beta, H, C, dt, method=method)

    B = 2

    T = 100

    xs = [torch.randn(B, 1, n)]

    for t in range(T):

        tinp = torch.tensor([t] * B).unsqueeze(1).to(dtype=torch.get_default_dtype())
        xs.append(sys.step(tinp, xs[-1]))

    x = torch.cat(xs, dim=1)

    for b in range(B):
        plt.subplot(B, 1, b + 1)
        plt.plot(range(T + 1), x[b, :, :].detach().numpy())

    plt.title('LorenzSystem')

    # plt.show()


if __name__ == '__main__':
    test()
