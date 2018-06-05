from math import pi
import torch
import torch.nn.functional as F

class GP_adapter(torch.nn.Module):

    def __init__(self, n_pivots=50, n_mc_samples=32, n_neuron_conv=8, n_neuron_fc=64, n_classes=2):
        
        """
        n_pivots: Number of regularly sampled points of the GP regression
        n_mc_samples: Number of GP posterior realizations to train the classifier
        n_neuron_conv: Number of filters of the conv layer
        n_neuron_hidden: Number of neurons of the FC hidden layer
        n_classes: Number of outputs
        """
        super(GP_adapter, self).__init__()
        
        self.dtype = torch.float32
        self.n_pivots = n_pivots
        self.n_mc_samples = n_mc_samples
        # Kernel parameters
        self.gp_logvar_kernel = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        self.gp_logtau_kernel = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        self.gp_logvar_likelihood = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        # Layers
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=n_neuron_conv, kernel_size=3, stride=1) 
        self.fc1 = torch.nn.Linear(n_neuron_conv*n_pivots/2,  n_neuron_fc)
        self.fc2 = torch.nn.Linear(n_neuron_fc,  n_classes)
    
    def periodic_kernel(self, x1, x2, P):
        return  torch.exp(self.gp_logvar_kernel -2*torch.exp(self.gp_logtau_kernel)*torch.sin(pi*(torch.t(x1) - x2)/P)**2)
    
    def GP_fit_posterior(self, mjd, mag, err, P, jitter=1e-5):
        # Kernel matrices
        reg_points = torch.unsqueeze(torch.linspace(start=0, end=10, steps=self.n_pivots), dim=0)
        mjd = torch.unsqueeze(mjd, dim=0)
        Ktt = self.periodic_kernel(mjd, mjd, P)  + torch.diag(err**2) + torch.exp(self.gp_logvar_likelihood)*torch.eye(mjd.shape[1])
        Ktx = self.periodic_kernel(mjd, reg_points, P) 
        Kxx = self.periodic_kernel(reg_points, reg_points, P)
        Ltt =  torch.potrf(Ktt, upper=False)  # Cholesky lower triangular 
        # posterior mean and covariance
        tmp1 = torch.t(torch.trtrs(Ktx, Ltt, upper=False)[0])
        tmp2 = torch.trtrs(torch.unsqueeze(mag, dim=1), Ltt, upper=False)[0]
        mu =torch.t(torch.mm(tmp1, tmp2))
        S = Kxx - torch.mm(tmp1, torch.t(tmp1) )
        R = torch.potrf(S +  jitter*torch.eye(self.n_pivots), upper=True)
        return mu, R, reg_points
    
    def sample_from_posterior(self, mu, R):
        eps = torch.randn([self.n_mc_samples, self.n_pivots], dtype=torch.float32)
        return torch.addmm(mu, eps, R)
    
    def forward(self, mjd, mag, err, P):
        # Gaussian process fit
        mu, R, _ = self.GP_fit_posterior(mjd, mag, err, P)
        # Sampling layer
        self.z = self.sample_from_posterior(mu, R)
        # Convolutional layers
        x = F.relu(self.conv1(torch.unsqueeze(self.z, dim=1)))
        x = F.max_pool1d(x, kernel_size=2, padding=1)
        x = x.view(32, -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

