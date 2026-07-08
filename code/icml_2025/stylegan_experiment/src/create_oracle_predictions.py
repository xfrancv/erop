import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

def load_regression_oracle(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    print(f"Loading Regression Oracle from {ckpt_path}...")
    model = models.resnet50(weights=None)
    # Regression head: [Mean, LogVariance]
    model.fc = nn.Linear(2048, 2) 
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model

class BenchmarkDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        self.img_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_paths.sort() 
        
        print(f"Found {len(self.img_paths)} images in {img_dir}")

    def __getitem__(self, index):
        path = self.img_paths[index]
        try:
            image = Image.open(path).convert('RGB')
        except:
            #print(f"Warning: Corrupt image {path}! This should not happen!")
            image = Image.open(self.img_paths[index+10]).convert('RGB')
            
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(path)

    def __len__(self):
        return len(self.img_paths)

def process_directory(model, directory, batch_size, device, seed):
    """
    Generates the Ground Truth Benchmark.
    
    Outputs:
    1. target_sample: The 'observed' label (y ~ N(mu, sigma)). Used for TRAINING.
    2. target_mean:   The True Mean (mu). Used for EVALUATION (Regret).
    3. target_std:    The True Aleatoric Uncertainty (sigma). Used for EVALUATION (Risk).
    """
    # Deterministic randomness for reproducible "noise"
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = BenchmarkDataset(directory, transform=tfm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    means_list = []
    sigmas_list = []
    samples_list = []
    filenames_list = []
    
    print(f"Processing {directory}...")
    
    with torch.no_grad():
        for images, fnames in tqdm(loader):
            images = images.to(device)
            
            # Forward Pass: [Batch, 2] -> (Mean, LogVar)
            outputs = model(images) 
            pred_mean = outputs[:, 0]
            pred_log_var = outputs[:, 1]
            
            # Convert LogVar to Std
            # sigma = exp(0.5 * log_var)
            pred_std = torch.exp(0.5 * pred_log_var)
            
            # Sample y ~ N(mu, sigma)
            # This generates the "noisy label" for training
            sampled_y = torch.normal(pred_mean, pred_std)
            
            means_list.append(pred_mean.cpu().numpy())
            sigmas_list.append(pred_std.cpu().numpy())
            samples_list.append(sampled_y.cpu().numpy())
            filenames_list.extend(fnames)
            
    return {
        'filenames': np.array(filenames_list),
        'target_mean': np.concatenate(means_list, axis=0),   # Ground Truth Mu
        'target_std': np.concatenate(sigmas_list, axis=0),   # Ground Truth Sigma
        'target_sample': np.concatenate(samples_list, axis=0) # Observed y
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Root data dir containing train/ and test/")
    parser.add_argument("--ckpt_path", type=str, default="src/age_resnet50_regression.pth", help="Path to age_resnet50_regression.pth")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Load Oracle
    model = load_regression_oracle(args.ckpt_path, device)
    
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    if os.path.exists(train_dir):
        results = process_directory(model, train_dir, args.batch_size, device, args.seed)
        out_path = os.path.join(train_dir, "predictions.npz")
        print(f"Saving Train to {out_path}...")
        np.savez_compressed(out_path, **results)
    
    if os.path.exists(test_dir):
        # Use seed+1 for distinct noise generation
        results = process_directory(model, test_dir, args.batch_size, device, args.seed + 1)
        out_path = os.path.join(test_dir, "predictions.npz")
        print(f"Saving Test to {out_path}...")
        np.savez_compressed(out_path, **results)
        
    print("Done. Predictions generated.")

if __name__ == "__main__":
    main()