import torch
import torch.nn as nn

class ParametricContinuousConvolution(nn.Module):
    def __init__(self, induction_points, kernel_neurons=8):
        super(type(self), self).__init__()
        self.kernel = nn.Sequential(nn.Linear(1, kernel_neurons), nn.Tanh(), nn.Linear(kernel_neurons, 1))
        self.y = induction_points

    def forward(self, light_curve):
        x, f = light_curve[:, 0, :], light_curve[:, 1, :].unsqueeze(1)
        dxy = torch.abs(x.unsqueeze(1) - self.y.repeat(x.shape[0], 1).unsqueeze(-1)).ravel().reshape(-1 ,1)
        return (self.kernel(dxy).reshape(x.shape[0], len(self.y), -1)*f).sum(dim=-1)
        #return (self.kernel(dxy).reshape(x.shape[0], -1, x.shape[-1])*f).sum(dim=1)


