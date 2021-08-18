# %% [markdown]
# - JIT?? https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
# - faster datasetdataloader
# - c++?
# - real data?
# - https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
# 
# 

# %%
%matplotlib ipympl
import numpy as np
import matplotlib.pyplot as plt
import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "2"


# %%
t = np.linspace(0, 4, num=100, dtype=np.float32)
t += np.random.randn(len(t))*0.01
t = np.sort(t)
P = 1.1234
m = np.sin(2.0*np.pi*t/P) + 0.5*np.sin(2.0*np.pi*2*t/P)  + 0.25*np.sin(2.0*np.pi*3*t/P)
m += np.random.randn(len(m))*0.2
fig, ax = plt.subplots(2)
ax[0].plot(t, m, '.')
ax[1].plot(np.mod(t, P)/P, m, '.')


# %%
import torch
import torch.nn as nn

class PeriodFinder(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 5, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 5, stride=1)
        self.conv3 = nn.Conv1d(16, 16, 5, stride=1)
        self.apool = nn.AdaptiveAvgPool1d(1)
        self.linear1 = nn.Linear(16, 1)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        h = self.activation(self.conv1(x))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        h = self.apool(h)
        return self.linear1(h.view(-1, self.linear1.weight.shape[1]))    


from torch.utils.data import Dataset, DataLoader

class lc_folder(Dataset):
    
    def __init__(self, mjd, mag):
        self.mjd =  torch.from_numpy(mjd.astype('float32'))
        self.mag =  torch.from_numpy(mag.astype('float32')).unsqueeze(0)
        self.freq = torch.arange(1e-4, 5, step=1e-4)
        
    def __getitem__(self, idx):
        phi = torch.remainder(self.mjd, 1/self.freq[idx])        
        return self.mag[:, torch.argsort(phi)]
    
    def __len__(self):
        return self.freq.shape[0]
    
class lc_trainer(Dataset):
    
    def __init__(self, mjd, mag, P):
        self.mjd =  torch.from_numpy(mjd.astype('float32'))
        self.mag =  torch.from_numpy(mag.astype('float32')).unsqueeze(0)
        self.P = P
        
    def __getitem__(self, idx):
        label = 0.
        if torch.rand(1) > 0.5:
            label = 1.
            phi = torch.remainder(self.mjd, self.P)
        else:
            phi = torch.remainder(self.mjd, 5*torch.rand(1)+1e-4)
        
        return self.mag[:, torch.argsort(phi)], label
    
    def __len__(self):
        return 1

data_train = lc_trainer(t, m, P)
data_eval = lc_folder(t, m)

with torch.jit.optimized_execution(True):
    my_script_module = torch.jit.script(PeriodFinder())

torch.set_num_threads(1)


# %%
model = PeriodFinder()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1000):
    epoch_loss = 0.0
    for folded_data, label in DataLoader(data_train, batch_size=1):
        optimizer.zero_grad()
        yhat = model(folded_data)
        loss = criterion(yhat.squeeze(0), label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(epoch_loss)


# %%
torch.rand(1) > 0.5


# %%
#%%timeit -r3 -n1
#%%prun
output = torch.tensor([])
with torch.no_grad():
    for folded_data in DataLoader(data_eval, batch_size=512, num_workers=1):
        output = torch.cat((output, model(folded_data)))


# %%
fig, ax = plt.subplots()
ax.plot(data_eval.freq.numpy(), nn.Sigmoid()(output).numpy()[:, 0])


# %%
get_ipython().run_cell_magic('timeit', '-r3 -n1', 'import P4J\nper = P4J.periodogram(method="MHAOV")\nper.set_data(t, m, m)\nper.frequency_grid_evaluation(fmin=0, fmax=5., fresolution=1e-4)')


# %%



