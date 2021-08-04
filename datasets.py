from os.path import join
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# wget http://faculty.washington.edu/ivezic/linear/PaperIII/allDAT.tar.gz
# wget http://faculty.washington.edu/ivezic/linear/PaperIII/PLV_LINEAR.dat
# tar xzvf allDAT.tar.gz

LINEAR_labels = {1: 'RR Lyrae ab', 2: 'RR Lyrae c', 3: 'Algol 1 minimum', 4: 'Algol 2 mininum', 5: 'Contact binary', 6: 'Delta Scuti', 7: 'LPV', 8: 'Hearbeat', 9: 'BL Her', 11: 'Anomalous Cepheid', 0: 'Other'}


class LINEAR(Dataset):

    def __init__(self, path: str='.', transform=None):
        # TODO: option to select a subset of the classes
        meta_file = join(path, "PLV_LINEAR.dat")
        data_path = join(path, 'allDAT/')
        df = pd.read_csv(meta_file, skiprows=32, sep='\s+',escapechar='#', index_col=0)        
        self.data, self.ids, self.labels, self.periods = [], [], [], []
        self.transform = transform
        for id, (label, period) in df[["LCtype", "P"]].iterrows():
            file_path = join(data_path, f"{id}.dat")
            self.data.append(pd.read_csv(file_path, delim_whitespace=True).values)
            self.ids.append(id)
            self.labels.append(label)
            self.periods.append(period)

    def __getitem__(self, idx: int) -> Tuple:
        sample = {'light_curve': self.data[idx], 'label': self.labels[idx], 'period': self.periods[idx]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.labels)

    def plot_lightcurve(self, idx: int):
        fig, ax = plt.subplots(1, 2, figsize=(6, 2), sharey=True, tight_layout=True)
        ax[0].invert_yaxis()
        mjd, mag, err = self.data[idx].T
        period = self.periods[idx]
        ax[0].errorbar(mjd, mag, err, fmt='.')
        ax[0].set_xlabel('Julian date')
        ax[0].set_ylabel('Magnitude')
        ax[0].set_title(f"ID: {self.ids[idx]}")
        ax[1].errorbar(np.mod(mjd, period)/period, mag, err, fmt='.')
        ax[1].set_xlabel('Phase')        
        ax[1].set_title(f"Label: {LINEAR_labels[self.labels[idx]]}")

if __name__ == "__main__":

    dataset = LINEAR()
    dataset.plot_lightcurve(0)
    fig, ax = plt.subplots(figsize=(6, 2), sharey=True, tight_layout=True)
    lc_data = dataset[0]['light_curve']
    ax.errorbar(lc_data[:, 0], lc_data[:, 1], lc_data[:, 2], fmt='.')


