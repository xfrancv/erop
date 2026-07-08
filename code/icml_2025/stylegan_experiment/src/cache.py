import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import ImageDataset

def extract_loop(model, loader, device):
    """
    Runs inference loop. 
    Returns: (Features, Observed_Y, Oracle_Sigma, Oracle_Mean)
    """
    model.eval()
    features_list = []
    y_list = []
    sigma_list = []
    mu_list = []
    
    with torch.no_grad():
        # Loader yields: image, y, sigma, mu
        for imgs, ys, sigmas, mus in tqdm(loader, desc="Extracting Features"):
            imgs = imgs.to(device)
            # Forward pass
            feats = model(imgs)
            
            features_list.append(feats.cpu().numpy())
            y_list.append(ys.numpy())
            sigma_list.append(sigmas.numpy())
            mu_list.append(mus.numpy())
            
    return (np.concatenate(features_list, axis=0), 
            np.concatenate(y_list, axis=0), 
            np.concatenate(sigma_list, axis=0),
            np.concatenate(mu_list, axis=0))

def get_cached_features(args, backbone_model, train_files, test_files, device):
    """
    Checks cache. If missing, extracts features for BOTH train and test sets 
    of the specific Fold and saves them.
    """
    cache_name = f"fold_{args.fold}_{args.backbone_name}.npz"
    cache_path = os.path.join(args.cache_dir, cache_name)
    
    if os.path.exists(cache_path) and not args.force_extract:
        print(f"[CACHE] Loading features from {cache_path}")
        data = np.load(cache_path)
        return (data['X_train'], data['y_train'], data['sigma_train'], data['mu_train'],
                data['X_test'], data['y_test'], data['sigma_test'], data['mu_test'])
    
    print(f"[CACHE] Cache miss for {cache_name}. extracting...")
    
    # 1. Setup Directories
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    labels_path = os.path.join(train_dir, "predictions.npz")
    test_labels_path = os.path.join(test_dir, "predictions.npz")
    
    # 2. Setup Datasets
    tfm = backbone_model.get_transform()
    
    train_ds = ImageDataset(train_dir, train_files, labels_path=labels_path, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    test_ds = ImageDataset(test_dir, test_files, labels_path=test_labels_path, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 3. Extract
    print("--- Extracting Train ---")
    X_train, y_train, s_train, m_train = extract_loop(backbone_model, train_loader, device)
    
    print("--- Extracting Test ---")
    X_test, y_test, s_test, m_test = extract_loop(backbone_model, test_loader, device)
    
    # 4. Save
    os.makedirs(args.cache_dir, exist_ok=True)
    print(f"[CACHE] Saving to {cache_path}")
    np.savez_compressed(
        cache_path,
        X_train=X_train, y_train=y_train, sigma_train=s_train, mu_train=m_train,
        X_test=X_test, y_test=y_test, sigma_test=s_test, mu_test=m_test
    )
    
    return X_train, y_train, s_train, m_train, X_test, y_test, s_test, m_test