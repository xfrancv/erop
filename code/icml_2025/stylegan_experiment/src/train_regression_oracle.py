import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

def compute_moments_from_probs(probs, ages):
    """
    Converts Softmax Probabilities [N, 102] into Mean [N] and Variance [N].
    """
    # E[x] = sum(p * x)
    mean = (probs * ages).sum(dim=1)
    
    # Var[x] = E[x^2] - (E[x])^2
    # This captures the Oracle's uncertainty
    variance = (probs * (ages - mean.unsqueeze(1))**2).sum(dim=1)
    
    return mean, variance

class DistillationDataset(Dataset):
    def __init__(self, image_dir, npz_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        print(f"Loading oracle predictions from {npz_path}...")
        data = np.load(npz_path)
        self.filenames = data['filenames']
        
        # Find the posterior key (usually 'age_posteriors' or 'fc_posteriors')
        keys = [k for k in data.keys() if 'posteriors' in k]
        if not keys:
            raise ValueError("No 'posteriors' key found in npz file.")
            
        # [N, 102]
        self.probs = data[keys[0]] 
        
        # Pre-compute targets (Mean and Log-Variance) to save training time
        print("Pre-computing Mean and Variance targets...")
        tensor_probs = torch.from_numpy(self.probs).float()
        ages = torch.arange(tensor_probs.shape[1]).float().unsqueeze(0)
        
        self.targets_mean, self.targets_var = compute_moments_from_probs(tensor_probs, ages)
        
        # Predict Log Variance for numerical stability
        self.targets_log_var = torch.log(self.targets_var + 1e-6)

        print(f"Loaded {len(self.filenames)} samples.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.image_dir, os.path.basename(fname))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Robustness: return black image if file missing/corrupt
            print(f"Warning: Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
            
        return image, self.targets_mean[idx], self.targets_log_var[idx]

def get_regression_model(ckpt_path, device):
    print("Building Regression Oracle...")
    # Initialize standard ResNet50
    model = models.resnet50(weights=None)
    
    # 1. Load Original Classifier Weights
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    print(f"Loading classifier weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Handle Architecture Mismatch (Sequential vs Linear head)
    has_sequential = any("fc.1.weight" in k for k in state_dict.keys())
    if has_sequential:
        model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 102))
    else:
        model.fc = nn.Linear(2048, 102)
        
    model.load_state_dict(state_dict)
    
    # 2. FREEZE BACKBONE
    # We want features identical to the classifier features
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. REPLACE HEAD FOR REGRESSION
    # Input: 2048 features
    # Output: 2 scalars (Mean, Log-Variance)
    model.fc = nn.Linear(2048, 2)
    
    model = model.to(device)
    return model

def train_oracle(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu")
    
    # Setup Data
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    full_dataset = DistillationDataset(args.image_dir, args.npz_path, transform=tfm)
    
    # Split 80/20 for Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Data Split: {len(train_dataset)} Train, {len(val_dataset)} Val")
    
    # Setup Model
    model = get_regression_model(args.ckpt_path, device)
    
    # Setup Optimization (Only optimize head)
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    print("Starting Linear Probing (Distillation)...")
    
    for epoch in range(args.epochs):
        model.train() # BatchNorm tracks stats, but backbone is frozen
        train_loss_mean = 0.0
        train_loss_var = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, tgt_mean, tgt_log_var in pbar:
            images = images.to(device)
            tgt_mean = tgt_mean.to(device)
            tgt_log_var = tgt_log_var.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images) # [B, 2]
            pred_mean = outputs[:, 0]
            pred_log_var = outputs[:, 1]
            
            # Loss Components
            l_mean = criterion(pred_mean, tgt_mean)
            l_var = criterion(pred_log_var, tgt_log_var)
            
            loss = l_mean + l_var
            loss.backward()
            optimizer.step()
            
            train_loss_mean += l_mean.item()
            train_loss_var += l_var.item()
            
            pbar.set_postfix({'L_mu': l_mean.item(), 'L_var': l_var.item()})

        avg_train_mean = train_loss_mean / len(train_loader)
        avg_train_var = train_loss_var / len(train_loader)

        model.eval()
        val_loss_mean = 0.0
        val_loss_var = 0.0
        
        with torch.no_grad():
            for images, tgt_mean, tgt_log_var in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = images.to(device)
                tgt_mean = tgt_mean.to(device)
                tgt_log_var = tgt_log_var.to(device)
                
                outputs = model(images)
                pred_mean = outputs[:, 0]
                pred_log_var = outputs[:, 1]
                
                l_mean = criterion(pred_mean, tgt_mean)
                l_var = criterion(pred_log_var, tgt_log_var)
                
                val_loss_mean += l_mean.item()
                val_loss_var += l_var.item()

        avg_val_mean = val_loss_mean / len(val_loader)
        avg_val_var = val_loss_var / len(val_loader)
        total_val_loss = avg_val_mean + avg_val_var
        
        print(f"\n[Epoch {epoch+1}] Train L_mu: {avg_train_mean:.4f} L_var: {avg_train_var:.4f} || "
              f"Val L_mu: {avg_val_mean:.4f} L_var: {avg_val_var:.4f}")
        
        # --- CHECKPOINTING ---
        if total_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} -> {total_val_loss:.4f}). Saving model...")
            best_val_loss = total_val_loss
            if args.output_path:
                torch.save(model.state_dict(), args.output_path)
        else:
            print("Validation loss did not improve.")
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to original images")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to predictions.npz (Soft labels)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to age_resnet50.pth (Classifier)")
    parser.add_argument("--output_path", type=str, default="age_resnet50_regression.pth")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10) 
    
    args = parser.parse_args()
    train_oracle(args)