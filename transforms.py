from typing import Dict, List
import numpy as np
import torch

class Compose:
    """
    Creates a sequence of transformations
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample: Dict[np.ndarray, int, float]) -> Dict[np.ndarray, int, float]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class PeriodFold:
    """
    Transforms the time axis of the light curve to phase using its period
    """
    def __init__(self, sort_phase: bool=True):
        self.sort_phase = sort_phase

    def __call__(self, sample: Dict[np.ndarray, int, float]) -> Dict[np.ndarray, int, float]:
        light_curve, period = sample['light_curve'], sample['period']
        light_curve[:, 0] = np.mod(light_curve[:, 0], period)/period
        if self.sort_phase:
            sample['light_curve'] = np.sort(light_curve, axis=0)
        else:
            sample['light_curve'] = light_curve
        return sample

class Normalize:
    """
    Centers the magnitudes. Rescales the magnitudes and the photometric errors
    """
    def __init__(self, robust: bool=True):
        self.robust = robust

    def __call__(self, sample: Dict[np.ndarray, int, float]) -> Dict[np.ndarray, int, float]:
        light_curve = sample['light_curve']
        if self.robust:
            center = np.median(light_curve[:, 1])
            scale = np.percentile(light_curve[:, 1], [25, 75])
            scale = scale[1] - scale[0]
        else:
            center = np.mean(light_curve[:, 1])
            scale = np.std(light_curve[:, 1])
        light_curve[:, 1] = (light_curve[:, 1] - center)/scale
        light_curve[:, 2] /= scale
        sample['light_curve'] = light_curve
        return sample

class Interpolate:

    def __init__(self, x: np.ndarray=np.linspace(0., 1., 30)):
        """
        TODO
        - improve naming
        - interpolate using errorbars
        - interpolate using kernels
        - propagate uncertainty
        """
        self.x = x

    def __call__(self, sample: Dict[np.ndarray, int, float]) -> Dict[np.ndarray, int, float]:
        light_curve = sample['light_curve']
        sample['light_curve'] = np.interp(self.x, light_curve[:, 0], light_curve[:, 1]).reshape(1, -1)
        return sample


class ToTensor:
    """
    Converts ``ndarray`` light curve to ``torch.Tensor``
    """

    def __call__(self, sample: Dict[np.ndarray, int, float]) -> Dict[torch.Tensor, int, float]:
         light_curve = sample['light_curve']
         sample['light_curve'] = torch.from_numpy(light_curve.astype('float32'))
         return sample

