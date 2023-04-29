import torch
from typing import Dict, List
import numpy as np
from torch.nn.functional import pad

class Collate_and_transform:
    """
    Creates a sequence of transformations
    TODO: ADD ZERO PAD
    """

    def __init__(self, transforms: List=[], pad: bool=False):
        self.transforms = transforms
        self.pad = pad

    def __call__(self, batch):
        transformed_batch = []
        if self.pad:
            lc_len = [sample['light_curve'].size(-1) for sample in batch]            
        for i, sample in enumerate(batch):
            if self.pad:
                sample['light_curve'] = pad(sample['light_curve'], (0, max(lc_len)-lc_len[i]))
            for transform in self.transforms:
                sample = transform(sample)
            transformed_batch.append(sample)
        if len(batch) > 1:
            data = torch.stack([sample['light_curve'] for sample in transformed_batch], dim=0)
        else:
            data = sample['light_curve'].unsqueeze(0)
        labels = torch.LongTensor([sample['label'] for sample in transformed_batch])
        return {'light_curve': data, 'label': labels}




class PeriodFold:
    """
    Transforms the time axis of the light curve to phase using its period
    """
    def __init__(self, sort_phase: bool=False):
        self.sort_phase = sort_phase

    def __call__(self, sample: Dict) -> Dict:
        light_curve, period = sample['light_curve'], sample['period']
        # Period folding transformation
        light_curve[0] = torch.remainder(light_curve[0], period)/period
        if self.sort_phase:
            light_curve = light_curve[:, torch.argsort(light_curve[0])]
        sample['light_curve'] = light_curve
        return sample

class RandomPeriodFold:
    """
    Transforms the time axis of the light curve to phase using its period
    """
    def __init__(self, sort_phase: bool=False, p=0.5):
        self.sort_phase = sort_phase
        self.p = p
        self.multiples = [1/4, 1/3, 1/2, 2, 3, 4]

    def __call__(self, sample: Dict) -> Dict:
        light_curve, period = sample['light_curve'], sample['period']
        if torch.bernoulli(torch.Tensor([self.p])).item():
            if torch.bernoulli(torch.Tensor([self.p])).item():
                period = 0.01+torch.rand(1)*100
            else:
                period = period*self.multiples[torch.randperm(6)[0]]
            sample['label'] = 0
        else:
            sample['label'] = 1
        
        # Period folding transformation
        light_curve[0] = torch.remainder(light_curve[0], period)/period
        if self.sort_phase:
            light_curve = light_curve[:, torch.argsort(light_curve[0])]
        sample['light_curve'] = light_curve
        return sample

class LinearInterpolation:

    def __init__(self, min=0., max=1., n_grid=30):
        self.hatx = torch.linspace(min, max, n_grid)
    
    def __call__(self, sample: Dict) -> Dict:
        x, y, _ = sample['light_curve']
        # TODO: Implementar en torch
        yhat = torch.from_numpy(np.interp(self.hatx.numpy(), x.numpy(), y.numpy()))
        sample['light_curve'] = yhat.unsqueeze(0)
        return sample

class KernelInterpolation:
    """TODO: More kernels
    """

    def __init__(self, min: float=0., max: float=1., n_grid: int=30):
        self.hatx = torch.linspace(min, max, n_grid)
    
    def __call__(self, sample: Dict) -> Dict:
        x, y, dy = sample['light_curve']
        w = torch.exp((torch.cos(2.0*3.14159*(self.hatx.reshape(-1,1) - x)) -1)/1**2)/dy**2
        norm = torch.sum(w, axis=1) # This can be very small!
        yhat = torch.sum(w*y, axis=1)/norm
        sample['light_curve'] = yhat.unsqueeze(0)
        return sample


if __name__ == "__main__":

    from datasets import LINEAR
    from transforms_datasets import ToTensor
    import matplotlib.pyplot as plt
    #from torch.utils.data import DataLoader
    dataset = LINEAR(path='.', transform=ToTensor())
    collator1 = Collate_and_transform([PeriodFold()])    
    #%matplotlib ipympl
    fig, ax = plt.subplots(4, figsize=(6, 6), sharey=True, tight_layout=True)
    mjd, mag, err = dataset[0]['light_curve']
    ax[0].errorbar(mjd, mag, err, fmt='.')
    phi, mag, err = collator1([dataset[0]])['light_curve'][0]
    ax[1].errorbar(phi, mag, err, fmt='.')
    for n_grid in [10, 100]:
        collator2 = Collate_and_transform([PeriodFold(sort_phase=True), LinearInterpolation(n_grid=n_grid)])
        collator3 = Collate_and_transform([PeriodFold(), KernelInterpolation(n_grid=n_grid)])
        lc_data = collator2([dataset[0]])['light_curve'][0]
        ax[2].plot(np.linspace(0,1,n_grid),lc_data[0])
        lc_data = collator3([dataset[0]])['light_curve'][0]
        ax[3].plot(np.linspace(0,1,n_grid),lc_data[0])
        