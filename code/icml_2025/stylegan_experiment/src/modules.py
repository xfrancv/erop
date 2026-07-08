import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from tqdm import tqdm

class LinearMeanVar(nn.Module):
    """
    A simple MLP that outputs [Mean, LogVariance]
    """
    def __init__(self, input_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, 2)
        
    def forward(self, x):
        out = self.layer(x)
        return out[:, 0], out[:, 1] # Mean, LogVar

def train_noise_model(X_train, y_train, device, val_split=0.1, epochs=1000, patience=30):
    """
    Trains the student model to estimate aleatoric uncertainty (MLE).
    Returns the trained model and the learned variances for train/test.
    """
    print("\n--- Training Noise Model (MLE) ---")
    
    # 1. Internal Validation Split
    num_samples = len(X_train)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split = int(num_samples * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    
    # 2. To Device (Float64 preferred for consistency with the rest of pipeline)
    X = torch.tensor(X_train, dtype=torch.float64).to(device)
    y = torch.tensor(y_train, dtype=torch.float64).to(device)
    
    Xt, yt = X[train_idx], y[train_idx]
    Xv, yv = X[val_idx], y[val_idx]
    
    model = LinearMeanVar(input_dim=X_train.shape[1]).double().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 3. Loop
    best_loss = float('inf')
    best_state = None
    pat_count = 0
    
    pbar = tqdm(range(epochs), desc="Fitting Noise")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        mu, logvar = model(Xt)
        precision = torch.exp(-logvar)
        # NLL Loss
        loss = 0.5 * precision * (yt - mu)**2 + 0.5 * logvar
        loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            mu_v, logvar_v = model(Xv)
            prec_v = torch.exp(-logvar_v)
            val_loss = 0.5 * prec_v * (yv - mu_v)**2 + 0.5 * logvar_v
            val_loss = val_loss.mean().item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            pat_count = 0
            pbar.set_postfix({'Val': f"{val_loss:.4f} (*)"})
        else:
            pat_count += 1
            pbar.set_postfix({'Val': f"{val_loss:.4f}"})
            
        if pat_count >= patience:
            break
            
    # 4. Restore Best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    
    return model

def predict_variance(model, X, device):
    """
    Returns sigma^2 (variance)
    """
    model.eval()
    t_X = torch.tensor(X, dtype=torch.float64).to(device)
    with torch.no_grad():
        _, logvar = model(t_X)
    return torch.exp(logvar).cpu().numpy()