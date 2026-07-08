import torch
import numpy as np

try:
    import gpytorch
    GPYTORCH_AVAILABLE = True
except ImportError:
    GPYTORCH_AVAILABLE = False

class Estimator:
    def fit(self, X, y, sigma_sq):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

class BayesianLinearRegression(Estimator):
    def __init__(self, device):
        self.device = device
        self.w_N = None
        self.S_N = None

    def fit(self, X, y, sigma_sq):
        """
        X: (N, D)
        y: (N,) - centered
        sigma_sq: (N,) - noise variance per sample
        """
        N, D = X.shape
        # Add Bias Column
        X_b = np.hstack([X, np.ones((N, 1))])
        
        Phi = torch.tensor(X_b, dtype=torch.float64).to(self.device)
        t = torch.tensor(y, dtype=torch.float64).to(self.device)
        sigma = torch.tensor(sigma_sq, dtype=torch.float64).to(self.device)
        
        # Precision matrix (beta) is diagonal inverse of noise
        beta_diag = 1.0 / (sigma + 1e-9)
        
        # Phi^T * Beta * Phi
        # Efficient: (Phi.T * beta) @ Phi
        XtBX = (Phi.T * beta_diag) @ Phi
        XtBy = (Phi.T * beta_diag) @ t
        
        # Prior alpha * I
        alpha = 1.0
        I = torch.eye(D + 1, dtype=torch.float64).to(self.device)
        
        A = XtBX + alpha * I
        self.S_N = torch.linalg.inv(A)
        self.w_N = self.S_N @ XtBy

    def predict(self, X):
        N = X.shape[0]
        X_b = np.hstack([X, np.ones((N, 1))])
        Phi = torch.tensor(X_b, dtype=torch.float64).to(self.device)
        
        # Mean
        pred_mean = (Phi @ self.w_N).cpu().numpy()
        
        # Epistemic Variance: x^T * S_N * x
        # Row-wise dot product
        term = Phi @ self.S_N
        pred_epi = torch.sum(term * Phi, dim=1).cpu().numpy()
        
        return pred_mean, pred_epi

if GPYTORCH_AVAILABLE:
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel_type='linear'):
            super().__init__(train_x, train_y, likelihood)
            if kernel_type == 'linear':
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.LinearKernel()
            elif kernel_type == 'rbf':
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class GPEstimator(Estimator):
        def __init__(self, device, kernel='linear'):
            self.device = device
            self.kernel = kernel
            self.model = None
            self.likelihood = None
            self.X_train = None

        def fit(self, X, y, sigma_sq):
            # In linear GP, we often add bias column manually or rely on constant mean.
            # GPyTorch LinearKernel usually implies x^T x. Let's add bias column for Linear.
            if self.kernel == 'linear':
                X = np.hstack([X, np.ones((X.shape[0], 1))])
            
            self.X_train = torch.tensor(X, dtype=torch.float64).to(self.device)
            t_y = torch.tensor(y, dtype=torch.float64).to(self.device)
            t_noise = torch.tensor(sigma_sq, dtype=torch.float64).to(self.device)
            
            # Fixed Noise Likelihood (Heteroscedastic)
            self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=t_noise, learn_additional_noise=False
            )
            
            self.model = ExactGPModel(self.X_train, t_y, self.likelihood, self.kernel).to(self.device)
            self.model.double() # Enforce float64
            
            # Initialize Hyperparams
            if self.kernel == 'linear':
                self.model.covar_module.variance = 1.0
            elif self.kernel == 'rbf':
                self.model.covar_module.base_kernel.lengthscale = 22.0 #45.0 # Heuristic initialization
                
            self.model.eval()
            self.likelihood.eval()

        def predict(self, X):
            if self.kernel == 'linear':
                X = np.hstack([X, np.ones((X.shape[0], 1))])
                
            t_X = torch.tensor(X, dtype=torch.float64).to(self.device)
            
            batch_size = 32
            means = []
            vars = []
            
            with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
                for i in range(0, len(t_X), batch_size):
                    batch = t_X[i:i+batch_size]
                    out = self.model(batch)
                    means.append(out.mean.cpu().numpy())
                    vars.append(out.variance.cpu().numpy())
                    
            return np.concatenate(means), np.concatenate(vars)