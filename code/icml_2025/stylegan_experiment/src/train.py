import argparse
import os
import json
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from src.backbones import load_backbone
from src.cache import get_cached_features
from src.preprocessing import Preprocessor
from src.modules import train_noise_model, predict_variance
from src.estimators import BayesianLinearRegression, GPEstimator

def main():
    parser = argparse.ArgumentParser()
    # Path Arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Root containing train/ and test/")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to ResNet .pth or FaRL .pth")
    parser.add_argument("--backbone_name", type=str, required=True, choices=["resnet", "farl"])
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--output_dir", type=str, default="results")
    
    # Experiment Config
    parser.add_argument("--estimator", type=str, default="blr", choices=["blr", "gp_linear", "gp_rbf"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--force_extract", action='store_true')
    parser.add_argument("--use_oracle_noise", action='store_true', help="Use Ground Truth sigma instead of learning it (Upper Bound)")
    
    args = parser.parse_args()
    
    # 1. Setup & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment: {args.backbone_name} + {args.estimator} | Fold {args.fold} | Frac {args.fraction}")
    
    # 2. Load Manifest
    manifest_path = os.path.join(args.data_dir, "benchmark_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    fold_str = str(args.fold)
    if fold_str not in manifest['folds']:
        raise ValueError(f"Fold {args.fold} not found in manifest.")
        
    train_files = manifest['folds'][fold_str]['train']
    test_files = manifest['test_set']
    
    # 3. Load/Extract Features (Full Fold)
    backbone = load_backbone(args.backbone_name, args.ckpt_path, device)
    
    # Unpack ALL 8 arrays
    (X_train_full, y_train_full, sigma_train_full, mu_train_full,
     X_test_full, y_test_full, sigma_test_full, mu_test_full) = get_cached_features(
         args, backbone, train_files, test_files, device
    )
    
    del backbone
    torch.cuda.empty_cache()

    # 4. Subsampling (The Fraction Logic)
    n_total = len(X_train_full)
    n_keep = int(n_total * args.fraction)
    
    rng = np.random.default_rng(args.seed + args.fold)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    keep_indices = indices[:n_keep]
    
    print(f"Subsampling: Using {n_keep}/{n_total} samples.")
    X_train = X_train_full[keep_indices]
    y_train = y_train_full[keep_indices]
    sigma_train_oracle = sigma_train_full[keep_indices]
    
    # 5. Preprocessing
    scaler = Preprocessor()
    scaler.fit(X_train, y_train)
    
    X_train_scaled, y_train_centered = scaler.transform(X_train, y_train)
    X_test_scaled, _ = scaler.transform(X_test_full) 
    
    # 6. Define Noise (Aleatoric Uncertainty)
    if args.use_oracle_noise:
        print("[Config] Using ORACLE noise (Ground Truth).")
        sigma_train_sq = sigma_train_oracle ** 2
        sigma_test_sq = sigma_test_full ** 2
    else:
        print("[Config] Learning STUDENT noise (MLE).")
        # Train the Noise Model on the available fraction
        noise_model = train_noise_model(X_train_scaled, y_train_centered, device)
        
        # Predict variance (sigma^2)
        sigma_train_sq = predict_variance(noise_model, X_train_scaled, device)
        sigma_test_sq = predict_variance(noise_model, X_test_scaled, device)
    
    # 7. Train Estimator (Epistemic)
    if args.estimator == "blr":
        est = BayesianLinearRegression(device)
    elif "gp" in args.estimator:
        kernel = args.estimator.split("_")[1] # 'linear' or 'rbf'
        est = GPEstimator(device, kernel=kernel)
        
    print(f"Fitting {args.estimator.upper()}...")
    est.fit(X_train_scaled, y_train_centered, sigma_train_sq)
    
    # 8. Predict
    mu_pred_centered, var_epi = est.predict(X_test_scaled)
    mu_pred = scaler.inverse_transform_y(mu_pred_centered)
    
    # 9. Save
    exp_name = f"{args.backbone_name}_{args.estimator}_fold{args.fold}_frac{args.fraction}"
    if args.use_oracle_noise:
        exp_name += "_oracle"

    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    out_path = os.path.join(save_dir, "predictions.npz")
    print(f"Saving results to {out_path}...")
    
    np.savez_compressed(
        out_path,
        filenames=np.array(test_files),
        regression_mean=mu_pred,
        var_epi_student=var_epi,
        var_ale_student=sigma_test_sq,
        oracle_mean=mu_test_full,
        oracle_sigma=sigma_test_full,
        fold=args.fold,
        fraction=args.fraction
    )
    print("Done.")

if __name__ == "__main__":
    main()