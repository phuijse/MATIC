import numpy as np
from torch.utils.data import Dataset

class SyntheticClass:
    def __init__(self, n_samples):
        self.n_samples = n_samples


class SyntheticDataset(Dataset):
    def __init__(self, generators,):
        
        self.light_curves = []
        self.labels = []
        self.metadata = []

        generator = lambda t, f: np.sin(2.0*np.pi*f*t)
        for i in range(n_samples):
            N = int(10 + 100*np.random.rand())
            P = 0.01 + np.random.rand()*100 
            s = 0.01 + np.random.rand()
            mjd = np.sort(np.random.rand(N)*100)
            mag = generator(mjd, 1./P)
            err = np.array([s]*N)
            mag += s*np.random.randn(N)
            self.light_curves.append(np.stack([mjd, mag, err]))
            self.metadata.append({'period': P})
            self.labels.append(0)


    def __getitem__(self, idx):
        return self.light_curves[idx], self.labels[idx], self.metadata[idx]

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":

    dataset = SinesDataset(1)
    light_curve, label, metadata = dataset[0]
    print(light_curve.shape)