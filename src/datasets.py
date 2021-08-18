from os.path import join
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# wget http://faculty.washington.edu/ivezic/linear/PaperIII/allDAT.tar.gz
# wget http://faculty.washington.edu/ivezic/linear/PaperIII/PLV_LINEAR.dat
# tar xzvf allDAT.tar.gz

LINEAR_class_names = {1: 'RR Lyrae ab', 2: 'RR Lyrae c', 3: 'Algol 1 minimum', 4: 'Algol 2 mininum', 5: 'Contact binary', 6: 'Delta Scuti', 7: 'LPV', 8: 'Hearbeat', 9: 'BL Her', 11: 'Anomalous Cepheid', 0: 'Other'}


class LINEAR(Dataset):

    def __init__(self, path: str, classes: Optional[List[int]]=None, transform=None):
        self.transform = transform
        meta_file = join(path, "PLV_LINEAR.dat")
        data_path = join(path, 'allDAT/')
        df = pd.read_csv(meta_file, skiprows=32, sep='\s+',escapechar='#', index_col=0)        
        if classes is not None:
            df = df.loc[df['LCtype'].isin(classes)]
        self.linear_labels = df['LCtype'].tolist()
        self.periods = df['P'].tolist()
        self.ids = df.index.tolist()
        self.data = []
        for id in self.ids:
            file_path = join(data_path, f"{id}.dat")
            self.data.append(pd.read_csv(file_path, delim_whitespace=True).values.T)
        class_frequency = Counter(self.linear_labels)
        self.n_classes = len(class_frequency)
        self.le = LabelEncoder().fit(list(class_frequency.keys()))
        self.labels = self.le.transform(self.linear_labels)
        

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, float]:
        sample = {'light_curve': self.data[idx], 'label': self.labels[idx], 'period': self.periods[idx]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.labels)

    def decode_labels(self, label: np.ndarray, return_names: bool=True) -> np.ndarray:
        linear_labels = self.le.inverse_transform(label)
        if return_names:
            return np.array([LINEAR_class_names[label] for label in linear_labels])
        else:
            return linear_labels

    def plot_lightcurve(self, idx: int, period: Optional[float]=None) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(6, 2), sharey=True, tight_layout=True)
        ax[0].invert_yaxis()
        mjd, mag, err = self.data[idx]
        if period is None:
            period = self.periods[idx]
        ax[0].errorbar(mjd, mag, err, fmt='.')
        ax[0].set_xlabel('Julian date')
        ax[0].set_ylabel('Magnitude')
        ax[0].set_title(f"ID: {self.ids[idx]}")
        ax[1].errorbar(np.mod(mjd, period)/period, mag, err, fmt='.')
        ax[1].set_xlabel('Phase')        
        ax[1].set_title(f"Class: {LINEAR_class_names[self.linear_labels[idx]]}")

if __name__ == "__main__":

    dataset = LINEAR(path='.')
    dataset.plot_lightcurve(0)
    fig, ax = plt.subplots(figsize=(6, 2), sharey=True, tight_layout=True)
    mjd, mag, err = dataset[0]['light_curve']
    ax.errorbar(mjd, mag, err, fmt='.')


