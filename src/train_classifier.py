import sys
import yaml
import json
import numpy as np
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, random_split

from datasets import LINEAR
from transforms_datasets import Compose, Normalize, ToTensor
from transforms_dataloader import Collate_and_transform, PeriodFold, KernelInterpolation
from models import SimpleConv

if len(sys.argv) != 4:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython train_classifier.py data_path model_path scores_path\n")
    sys.exit(1)

data_path = sys.argv[1]
model_path = sys.argv[2]
scores_path = sys.argv[3]

params = yaml.safe_load(open("params.yaml"))
dataset_seed = params["dataset_seed"]
lr = params["lr"]
n_grid = params["n_grid"]
nepochs = params["nepochs"]

data = LINEAR(path=data_path, classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
train_subset, valid_subset = random_split(data, (4000, len(data)-4000), generator=torch.Generator().manual_seed(dataset_seed))
collator = Collate_and_transform([PeriodFold(), KernelInterpolation(n_grid=n_grid)])
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collator)
valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=collator)

criterion = torch.nn.CrossEntropyLoss()
model = SimpleConv(n_channels=1, n_classes=data.n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for n in range(nepochs):
    global_loss = 0.0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()
        y = model.forward(batch['light_curve'])
        loss = criterion(y, batch['label'])
        loss.backward()
        optimizer.step()
        global_loss += loss.item()
    print(f"{n} {global_loss}")

torch.save(model, model_path)

preds = []
label = []    
with torch.no_grad():
    model.eval()
    for batch in valid_loader:
        preds.append(model.forward(batch['light_curve']).argmax(dim=1).numpy())
        label.append(batch['label'].numpy())
label = np.concatenate(label)
preds = np.concatenate(preds)
avg_prec = metrics.average_precision_score(label, preds)
roc_auc = metrics.roc_auc_score(label, preds)

with open(scores_path, "w") as fd:
    json.dump({"avg_prec": avg_prec, "roc_auc": roc_auc}, fd, indent=4)

print(metrics.classification_report(label, preds, target_names=data.decode_labels(np.unique(label))))
