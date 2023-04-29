import torch
from torch.utils.data import DataLoader, random_split

from ..models import ParametricContinuousConvolution
from ..datasets import LINEAR
from ..transforms.datasets import Compose, Normalize, ToTensor
from ..transforms.dataloader import Collate_and_transform

def test_pconv():
    layer = ParametricContinuousConvolution(induction_points=torch.arange(0, 1, step=0.1))
    dataset_seed = 1234
    data = LINEAR(path='.', classes=[1, 5], transform=Compose([Normalize(), ToTensor()]))
    train_subset, valid_subset = random_split(data, (4000, len(data)-4000), generator=torch.Generator().manual_seed(dataset_seed))
    collator = Collate_and_transform(pad=True)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collator)

    for batch in train_loader:
        break
    layer(batch['light_curve'])