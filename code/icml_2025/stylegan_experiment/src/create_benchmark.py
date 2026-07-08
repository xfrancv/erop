import argparse
import os
import glob
import json
import numpy as np

def create_manifest(args):
    print(f"Scanning {args.data_dir}...")
    
    # 1. Identify Files
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    
    # Check if directories exist to avoid errors
    if not os.path.exists(train_dir):
        print(f"Error: Train directory not found at {train_dir}")
        return
        
    train_files = sorted([os.path.basename(x) for x in glob.glob(os.path.join(train_dir, "*.png"))])
    
    # Check if test directory exists, handle gracefully if not
    if os.path.exists(test_dir):
        test_files = sorted([os.path.basename(x) for x in glob.glob(os.path.join(test_dir, "*.png"))])
    else:
        test_files = []
        print("Warning: No test directory found. Test set will be empty.")
    
    print(f"Found {len(train_files)} Total available images.")
    
    # 2. Global Shuffle
    # We mix everything up first so the partitions are random
    rng = np.random.default_rng(args.seed)
    train_files = np.array(train_files)
    rng.shuffle(train_files)
    
    # 3. Create Partitions (Disjoint Sets)
    # This ensures N splits that do not overlap
    fold_chunks = np.array_split(train_files, args.num_folds)
    
    manifest = {
        "metadata": {
            "type": "regression_benchmark_disjoint",
            "description": f"{args.num_folds} folds. Each fold uses completely unique data (no overlap between folds)."
        },
        "test_set": test_files,
        "folds": {}
    }
    
    print(f"\nGenerating {args.num_folds} Disjoint Folds...")
    
    for fold_idx in range(args.num_folds):
        # This chunk is the ONLY data this fold will ever see
        current_fold_files = fold_chunks[fold_idx]
        
        # Internal Split: 80% Train, 20% Val
        # If you want 100% train and 0% val, change 0.8 to 1.0
        n_total = len(current_fold_files)
        n_train = int(n_total * 0.8)
        
        # Since we already globally shuffled, we can just slice
        train_subset = current_fold_files[:n_train]
        val_subset = current_fold_files[n_train:]
        
        # 4. Structure Update
        # Direct list assignment. No fraction keys.
        manifest["folds"][str(fold_idx)] = {
            "train": train_subset.tolist(),
            "validation": val_subset.tolist()
        }
        
        print(f"Fold {fold_idx}: Total={n_total} | Train={len(train_subset)} | Val={len(val_subset)}")

    # 5. Save
    out_path = os.path.join(args.data_dir, "benchmark_manifest.json")
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nManifest saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of disjoint splits")
    parser.add_argument("--seed", type=int, default=42)
    
    # 'fractions' argument removed
    
    args = parser.parse_args()
    create_manifest(args)