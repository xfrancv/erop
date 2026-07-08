import numpy as np
from PIL import Image
from collections import defaultdict
from bbox_utils import crop_bbox
from utils import MyYamlLoader, create_dir, build_data_transform
from tqdm import tqdm
from typing import List, Callable
import albumentations
import argparse
import cv2
import json
import os
import torch
import torch.utils.data as data
import yaml
import glob

# The shared 8-point bbox here: 
# [top_left_col, top_left_row, top_right_col, top_right_row, 
#  bot_right_col, bot_right_row, bot_left_col, bot_left_row]
FIXED_BBOX = [-1, -1, 255, -1, 255, 255, -1, 255] 
# ==========================================

class DirectoryDataset(data.Dataset):
    """
    Scans a directory for images and applies a fixed bounding box to all of them.
    """
    def __init__(self,
                 img_root_folder: str,
                 input_size: List[int],
                 bbox_margin: List[float],
                 transform: Callable) -> object:
        
        self.input_size = input_size
        self.bbox_margin = bbox_margin
        self.transform = transform
        self.img_root_folder = img_root_folder

        # Find all images (png, jpg, jpeg)
        self.img_paths = []
        extensions = ['*.png']
        for ext in extensions:
            self.img_paths.extend(glob.glob(os.path.join(img_root_folder, ext)))
        
        self.img_paths.sort() # Ensure deterministic order
        self.num_images = len(self.img_paths)
        
        print(f"Found {self.num_images} images in {img_root_folder}")

    def __getitem__(self, index):
        path = self.img_paths[index]
        in_img = cv2.imread(path)
        
        # crop_bbox might fail if image load failed
        if in_img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Use the global FIXED_BBOX
        sample_img_cv, _ = crop_bbox(
            in_img, FIXED_BBOX, self.input_size, margin=self.bbox_margin, one_based_bbox=True)
        
        sample_img_pil = Image.fromarray(cv2.cvtColor(sample_img_cv, cv2.COLOR_BGR2RGB))
        sample_img_nn = self.transform(sample_img_pil)

        return sample_img_nn, os.path.basename(path)

    def __len__(self):
        return self.num_images


class Inference:
    def __init__(self, model_file: str, device=None):
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

        self.model = torch.jit.load(model_file, map_location=self.device)
        self.model.eval()

        extra_files = {'config': '', 'albumentations': ''}
        torch.jit.load(model_file, _extra_files=extra_files, map_location=self.device)

        self.config = yaml.load(extra_files['config'], Loader=MyYamlLoader)
        self.input_size = self.config['model']['input_size']
        self.bbox_margin = self.config['preprocess']['bbox_extension']

        self.transform = build_data_transform(
            albumentations.from_dict(json.loads(extra_files['albumentations'])))

    def predict_from_directory(self,
                               img_root_folder: str,
                               batch_size: int = 50,
                               num_workers: int = 4):
        
        dataset = DirectoryDataset(img_root_folder=img_root_folder,
                                   input_size=self.input_size,
                                   bbox_margin=self.bbox_margin,
                                   transform=self.transform)

        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

        # Accumulators
        self.posteriors = defaultdict(list)
        self.img_paths = []

        print("Running inference...")
        for inputs, filenames in tqdm(data_loader):
            inputs = inputs.to(self.device)
            self.img_paths.extend(filenames)

            with torch.no_grad():
                # Get risks, labels, AND posteriors
                _, _, batch_posteriors = self.model.get_prediction(inputs)

                for head, tensor in batch_posteriors.items():
                    # Move to CPU and add to list. 
                    # Tensor shape is usually [Batch, Num_Classes]
                    self.posteriors[head].append(tensor.cpu().numpy())

        # Concatenate batches
        results = {'filenames': np.array(self.img_paths)}
        
        for head, data_list in self.posteriors.items():
            # Stack into a single array: [Total_Images, Num_Classes]
            concat_data = np.concatenate(data_list, axis=0)
            results[f'{head}_posteriors'] = concat_data
            print(f"Head '{head}' shape: {concat_data.shape}")

        return results

def run_prediction(args):
    predictor = Inference(args.model, args.device)

    # Determine output path
    if args.output is None:
        # Default to same dir as images, but with .npz extension
        output_file = os.path.join(args.image_dir, "predictions.npz")
    else:
        output_file = args.output

    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Exiting.")
        return

    if len(os.path.dirname(output_file)) > 0:
        create_dir(os.path.dirname(output_file))

    # Run inference
    results = predictor.predict_from_directory(
        args.image_dir, 
        batch_size=args.batch_size
    )

    # Save to NPZ
    print(f"Saving results to {output_file}...")
    np.savez_compressed(output_file, **results)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicts attributes and saves posteriors to NPZ.")
    parser.add_argument("model", type=str, help="Path to the PyTorch model file (.jit)")
    parser.add_argument("image_dir", type=str, help="Path to the directory containing the input images")
    parser.add_argument("--output", default=None, type=str, help="Output .npz file path.")
    parser.add_argument("--device", type=int, default=None, help="Device ID (default: autodetect)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    
    args = parser.parse_args()
    
    run_prediction(args)