import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from datasets import LINEAR
from transforms_datasets import Compose, Normalize, ToTensor
from transforms_dataloader import Collate_and_transform, PeriodFold, KernelInterpolation
from models import SimpleConv

data = LINEAR(path='.', classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
train_subset, valid_subset = random_split(data, (4500, len(data)-4500), generator=torch.Generator().manual_seed(1234))
model = SimpleConv(n_classes=data.n_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_collator = Collate_and_transform([PeriodFold(), KernelInterpolation()])
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=train_collator)
valid_loader = DataLoader(valid_subset, batch_size=256, collate_fn=train_collator)

for n in range(10):
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

preds = []
label = []    
with torch.no_grad():
    model.eval()
    for batch in valid_loader:
        preds.append(model.forward(batch['light_curve']).argmax(dim=1).numpy())
        label.append(batch['label'].numpy())
from sklearn.metrics import classification_report
label = np.concatenate(label)
preds = np.concatenate(preds)
print(classification_report(label, preds, target_names=data.decode_labels(np.unique(label))))
