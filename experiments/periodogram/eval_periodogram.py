import yaml
import json
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, random_split

from datasets import LINEAR
from transforms_datasets import Compose, Normalize, ToTensor
from transforms_dataloader import Collate_and_transform, RandomPeriodFold, KernelInterpolation
from models import SimpleConv

params = yaml.safe_load(open("exp_period/params.yaml"))
dataset_seed = params["dataset_seed"]
lr = params["lr"]
n_grid = params["n_grid"]
nepochs = params["nepochs"]

data = LINEAR(path='.', classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
train_subset, valid_subset = random_split(data, (4000, len(data)-4000), generator=torch.Generator().manual_seed(dataset_seed))

#model = SimpleConv(n_channels=1, n_classes=1)
model = torch.load("exp_period/model_periodogram.pt")


min_freq = 0.01
max_freq = 2.0
delta_freq = 0.001
freqs = torch.arange(min_freq, max_freq, delta_freq)
interpolator = KernelInterpolation(n_grid=n_grid)
p = []
sig = torch.nn.Sigmoid()
with torch.no_grad():
    for freq in freqs:
        example = valid_subset[0]
        example['light_curve'][0] = torch.remainder(example['light_curve'][0], 1/freq)*freq
        example = interpolator(example)
        p.append(sig(model.forward(example['light_curve'].unsqueeze(0))).item())

%matplotlib ipympl
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(freqs, p)
ax.axvline(1/valid_subset[0]['period'], ls='--', c='r')



