#
# File: models.py
#
import torch

from ceem.nn import LNMLP

DEFAULT_NN_KWARGS = lambda H: dict(input_size=10*H, 
                    hidden_sizes=[32]*8,  
                    output_size = 6,
                    activation='tanh', gain=1.0, ln=False)

class LagModel(torch.nn.Module):
    def __init__(self, neural_net_kwargs, H):
        super().__init__()
        self.net = LNMLP(**neural_net_kwargs)
    
    def forward(self, data_batch):
        # data batch is size ntrajs * H * ndim. 
        # Controls are the first 4 dimensions of ndim, then 3 velocities, then 3 rotation rates
        ntrajs = data_batch.shape[0]
        state_t = data_batch[:, -1:, 4:]
        pred = self.net(data_batch.view(ntrajs, -1))
        return pred