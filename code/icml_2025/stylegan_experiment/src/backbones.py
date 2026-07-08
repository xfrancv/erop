import torch
import torch.nn as nn
from torchvision import models, transforms
import os

try:
    import clip
except ImportError:
    clip = None

class FeatureExtractor(nn.Module):
    def get_transform(self):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError

class ResNetBackbone(FeatureExtractor):
    def __init__(self, ckpt_path, device):
        super().__init__()
        print(f"Loading ResNet from {ckpt_path}...")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            
        # Load standard ResNet
        self.model = models.resnet50(weights=None)
        # Handle the specific checkpoint structure (which likely has a 2-unit FC layer)
        self.model.fc = nn.Linear(2048, 2)
        state_dict = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state_dict)
        
        # Replace FC with Identity to extract features
        self.model.fc = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.model(x)

class FaRLBackbone(FeatureExtractor):
    def __init__(self, farl_path, device):
        super().__init__()
        if clip is None:
            raise ImportError("CLIP is not installed. Please install it to use FaRL.")
            
        print(f"Loading FaRL from {farl_path}...")
        if not os.path.exists(farl_path):
            raise FileNotFoundError(f"FaRL weights not found at {farl_path}")

        # Load CLIP architecture
        model, self.preprocess = clip.load("ViT-B/16", device="cpu")
        
        # Load FaRL weights
        farl_state = torch.load(farl_path, map_location="cpu")
        model.load_state_dict(farl_state["state_dict"], strict=False)
        
        self.model = model.visual.float().to(device)
        self.model.eval()

    def get_transform(self):
        # FaRL/CLIP has its own specific preprocessing (bicubic resize, specific means)
        # We rely on the clip.load() returned transform, but we might need to ensure
        # it outputs a tensor if the dataset class doesn't handle it.
        return self.preprocess

    def forward(self, x):
        # Ensure correct size if not already
        if x.shape[-1] != 224 or x.shape[-2] != 224:
             x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', antialias=True)
        return self.model(x.float())

def load_backbone(name, path, device):
    if name.lower() == 'resnet':
        return ResNetBackbone(path, device)
    elif name.lower() == 'farl':
        return FaRLBackbone(path, device)
    else:
        raise ValueError(f"Unknown backbone: {name}")