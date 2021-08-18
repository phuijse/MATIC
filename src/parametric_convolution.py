import torch.nn as nn

class ParametricContinuousConvolution(nn.Module):
    def __init__(self, induction_points, kernel_neurons=8):
        super(type(self), self).__init__()
        self.kernel = nn.Sequential(nn.Linear(1, kernel_neurons), nn.Tanh(), nn.Linear(kernel_neurons, 1))
        self.y = induction_points

    def forward(self, light_curve):
        x, f = light_curve[:, 0, :], light_curve[:, 1, :].unsqueeze(1)
        dxy = torch.abs(x.unsqueeze(1) - self.y.repeat(x.shape[0], 1).unsqueeze(-1)).ravel().reshape(-1 ,1)
        #print(self.kernel(dxy.reshape(-1, x.shape[-1]*self.y.shape)))
        return (self.kernel(dxy).reshape(x.shape[0], len(self.y), -1)*f).sum(dim=-1)
        #return (self.kernel(dxy).reshape(x.shape[0], -1, x.shape[-1])*f).sum(dim=1)


import torch

layer = ParametricContinuousConvolution(induction_points=torch.arange(0, 1, step=0.1))


from datasets import LINEAR
from transforms_datasets import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, random_split
from transforms_dataloader import Collate_and_transform
import yaml
params = yaml.safe_load(open("exp_period/params.yaml"))
dataset_seed = params["dataset_seed"]
data = LINEAR(path='.', classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
train_subset, valid_subset = random_split(data, (4000, len(data)-4000), generator=torch.Generator().manual_seed(dataset_seed))
collator = Collate_and_transform(pad=True)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collator)

for batch in train_loader:
    break


layer(batch['light_curve'])