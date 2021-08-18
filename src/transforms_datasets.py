from typing import Dict, List
import numpy as np
import torch

class Compose:
    """
    Creates a sequence of transformations
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict) -> Dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Normalize:
    """
    Centers the magnitudes. Rescales the magnitudes and the photometric errors
    """
    
    def __init__(self, robust: bool=True):
        self.robust = robust

    def __call__(self, sample: Dict) -> Dict:
        mag = sample['light_curve'][1]
        if self.robust:
            center = np.median(mag)
            # Interquartile range
            scale = np.subtract(*np.percentile(mag, [75, 25]))            
        else:
            center = np.mean(mag)
            scale = np.std(mag)
        sample['light_curve'][1] = (mag - center)/scale
        sample['light_curve'][2] /= scale
        return sample


class ToTensor:
    """
    Converts ``ndarray`` light curve to ``torch.Tensor``
    """

    def __call__(self, sample: Dict) -> Dict:
         light_curve = sample['light_curve']
         sample['light_curve'] = torch.from_numpy(light_curve.astype('float32'))
         return sample



