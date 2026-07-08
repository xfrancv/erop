import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, img_dir, file_list, labels_path=None, transform=None):
        """
        img_dir: Root directory for images (e.g., data/train)
        file_list: List of filenames to load.
        labels_path: Path to the Oracle predictions.npz (optional).
        transform: Torchvision transform.
        """
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform
        
        self.y_map = {}     # Observed Label (Sampled)
        self.std_map = {}   # True Aleatoric Std
        self.mu_map = {}    # True Mean (Oracle Ground Truth)
        self.has_labels = False

        if labels_path:
            if os.path.exists(labels_path):
                print(f"Loading labels from {labels_path}")
                data = np.load(labels_path)
                
                # We assume the .npz has: 'filenames', 'target_sample', 'target_std', 'target_mean'
                fnames = data['filenames']
                samples = data['target_sample']
                stds = data['target_std']
                means = data['target_mean']
                
                # Build hash maps for O(1) lookup
                # keys are basenames (e.g. '001.png')
                self.y_map = {os.path.basename(f): y for f, y in zip(fnames, samples)}
                self.std_map = {os.path.basename(f): s for f, s in zip(fnames, stds)}
                self.mu_map = {os.path.basename(f): m for f, m in zip(fnames, means)}
                
                self.has_labels = True
                
                # Filter file_list to valid intersection
                original_count = len(self.file_list)
                self.file_list = [f for f in self.file_list if os.path.basename(f) in self.y_map]
                if len(self.file_list) < original_count:
                    print(f"Warning: Dropped {original_count - len(self.file_list)} files missing from label file.")
            else:
                print(f"Warning: Labels file {labels_path} not found. Proceeding without labels.")

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        basename = os.path.basename(fname)
        path = os.path.join(self.img_dir, basename)

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            # Fallback for corrupt images
            print(f"Error loading {path}: {e}")
            fallback_idx = (idx + 1) % len(self.file_list)
            path = os.path.join(self.img_dir, os.path.basename(self.file_list[fallback_idx]))
            image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        y = self.y_map[basename]
        sigma = self.std_map[basename]
        mu = self.mu_map[basename]
        
        # Returns: Image Tensor, Observed Target, True Sigma, True Mean
        return image, y, sigma, mu

    def __len__(self):
        return len(self.file_list)