from math import pi
import torch
import torch.nn.functional as F

class GP_adapter(torch.nn.Module):

    def __init__(self, n_pivots=50, n_mc_samples=32, n_neuron_conv=8, kernel_size=5, n_classes=2):
        
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
        self.kernel = "periodic"
        # Kernel parameters
        self.gp_logvar_kernel = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        self.gp_logtau_kernel = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        self.gp_logvar_likelihood = torch.nn.Parameter(0.1*torch.randn(1, dtype=self.dtype))
        # Layers
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=n_neuron_conv, kernel_size=kernel_size, 
                                     stride=1, bias=True) 
        self.conv2 = torch.nn.Conv1d(in_channels=n_neuron_conv, out_channels=n_neuron_conv, kernel_size=kernel_size, 
                                     stride=1, bias=True) 
        self.conv3 = torch.nn.Conv1d(in_channels=n_neuron_conv, out_channels=n_neuron_conv, kernel_size=kernel_size, 
                                     stride=1, bias=True) 
        self.gpool = torch.nn.AvgPool1d(kernel_size=n_pivots-3*(kernel_size-1), stride=1, padding=0)
        self.fc1 = torch.nn.Linear(in_features=n_neuron_conv, out_features=n_classes, bias=True)
    
    def stationary_kernel(self, x1, x2, non_trainable_params=None):
        # Gaussian kernel
        if self.kernel == "periodic" and non_trainable_params is not None:
            P = non_trainable_params['period']
            K = torch.exp(-2.0*torch.pow(torch.sin(pi*(torch.t(x1) - x2)/P), 2.0)*torch.exp(self.gp_logtau_kernel))
        elif self.kernel == "RBF":  # aka Gaussian, Square Exponential
            K = torch.exp(-0.5*torch.pow(torch.t(x1) - x2, 2.0)*torch.exp(self.gp_logtau_kernel))
        else:
            raise ValueError("Wrong kernel")
        return torch.exp(self.gp_logvar_kernel)*K
    
    def GP_fit_posterior(self, mjd, mag, err, P, end=1.0, jitter=1e-5):
        """
        Expect a time series sampled at *mjd* instants (t) with values *mag* (m) and associated errors *err* (s)
        
        Returns the posterior mean and factorized covariance matrix of the GP sampled at instants x
        \[
        \mu = K_{xt} (K_{tt} + \sigma^2 I + \text{diag}(s^2))^{-1} m,
        \]
        \[
        \Sigma = K_{xx} - K_{xt} (K_{tt} + \sigma^2 I)^{-1} K_{xt}^T + \sigma^2 I
        \]
        where $\sigma^2$ is the variance of the noise.
        """
        # Kernel matrices
        non_trainable_kparams = {'period': 1.0}
        reg_points = torch.unsqueeze(torch.linspace(start=0.0, end=1.0-1.0/self.n_pivots, 
                                                    steps=self.n_pivots), dim=0)
        mjd = torch.unsqueeze(mjd, dim=0)
        Ktt = self.stationary_kernel(mjd, mjd, non_trainable_kparams)
        Ktt += torch.diag(err**2) + torch.exp(self.gp_logvar_likelihood)*torch.eye(mjd.shape[1])
        Ktx = self.stationary_kernel(mjd, reg_points, non_trainable_kparams) 
        Kxx = self.stationary_kernel(reg_points, reg_points, non_trainable_kparams)
        Ltt =  torch.potrf(Ktt, upper=False)  # Cholesky lower triangular 
        # posterior mean and covariance
        tmp1 = torch.t(torch.trtrs(Ktx, Ltt, upper=False)[0])
        tmp2 = torch.trtrs(torch.unsqueeze(mag, dim=1), Ltt, upper=False)[0]
        mu =torch.t(torch.mm(tmp1, tmp2))
        S = Kxx - torch.mm(tmp1, torch.t(tmp1)) #+ torch.exp(self.gp_logvar_likelihood)*torch.eye(self.n_pivots)
        R = torch.potrf(S + jitter*torch.eye(self.n_pivots), upper=True)
        return mu, R, reg_points
    
    def sample_from_posterior(self, mu, R, n_samples=None):
        if n_samples is None:
            n_samples = self.n_mc_samples
        eps = torch.randn([n_samples, self.n_pivots], dtype=self.dtype)
        return torch.addmm(mu, eps, R)
    
    
    def forward(self, sample, return_gp=False):
        # Parse sample
        lcdata, P = sample['data'], sample['period']
        if len(lcdata.shape) == 3:  # Minibatch dim
            lcdata = lcdata[0]
            P = P[0]
        mjd = torch.remainder(lcdata[:, 0], P)/P
        mag = lcdata[:, 1]
        err = lcdata[:, 2]
        # Gaussian process fit
        mu, R, reg_points = self.GP_fit_posterior(mjd, mag, err, P)        
        # Sampling layer
        z = self.sample_from_posterior(mu, R)
        ## Repeat first two points at the end, cyclic conv, TODO: Find smarter way to do this
        # z = z.repeat(1, 2)[:, :self.n_pivots+3-1]  
        # Fully convolutional architecture
        z = torch.unsqueeze(z, dim=1)
        x = F.relu(self.conv1(z))        
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Global average pool
        x = self.gpool(x)
        x = x.view(self.n_mc_samples, -1)
        # Fully connected layers
        x = F.log_softmax(self.fc1(x), dim=1)
        if not return_gp:
            return x
        else:
            return x, (mu, R, reg_points)
    
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

