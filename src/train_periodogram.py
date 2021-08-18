
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
import dvclive
#dvclive.init()

params = yaml.safe_load(open("params.yaml"))
dataset_seed = params["dataset_seed"]
lr = params["lr"]
n_grid = params["n_grid"]
nepochs = params["nepochs"]

data = LINEAR(path='../', classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
train_subset, valid_subset = random_split(data, (4000, len(data)-4000), generator=torch.Generator().manual_seed(dataset_seed))
collator = Collate_and_transform([RandomPeriodFold(), KernelInterpolation(n_grid=n_grid)])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collator)
valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=collator)

model = SimpleConv(n_channels=1, n_classes=1)
criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for n in range(nepochs):
    
    global_loss = 0.0
    model.train()
    for batch in train_loader:        
        optimizer.zero_grad()
        y = model.forward(batch['light_curve'])
        loss = criterion(y.squeeze(1), batch['label'].float())
        loss.backward()
        optimizer.step()
        global_loss += loss.item()
    dvclive.log('train/loss', global_loss/len(train_subset))
    
    global_loss = 0.0
    model.eval()
    for batch in valid_loader:        
        y = model.forward(batch['light_curve'])
        loss = criterion(y.squeeze(1), batch['label'].float())
        global_loss += loss.item()
    dvclive.log('valid/loss', global_loss/len(valid_subset))
    dvclive.next_step()
        

torch.save(model, "model_periodogram.pt")

