# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Modified for high-throughput generation (Batched + Async I/O + GPU Resize)

import os
import re
import time
from typing import List, Optional, Tuple, Union
import concurrent.futures

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def save_image_task(img_np, path):
    """Worker function to save image."""
    PIL.Image.fromarray(img_np, 'RGB').save(path, compress_level=1) # Faster saving, slightly larger file

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--batch-size', help='Batch size', type=int, default=32, show_default=True)
@click.option('--resize', help='Resize output to WxH (e.g. 256)', type=int, default=256, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    batch_size: int,
    resize: Optional[int]
):
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    if hasattr(G.synthesis, 'input'):
        m = make_transform(translate, rotate)
        m = np.linalg.inv(m)
        G.synthesis.input.transform.copy_(torch.from_numpy(m))

    label_one = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label_one[:, class_idx] = 1

    num_batches = (len(seeds) + batch_size - 1) // batch_size
    
    # Thread pool for saving images in background
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    
    # Pre-allocate z buffers on CPU to avoid allocating inside loop
    # (Minor optimization)
    
    start_time = time.time()
    
    for batch_idx in range(num_batches):
        b_start = batch_idx * batch_size
        b_end = min((batch_idx + 1) * batch_size, len(seeds))
        batch_seeds = seeds[b_start:b_end]
        current_bs = len(batch_seeds)

        if batch_idx % 10 == 0:
            print(f'Generating batch {batch_idx+1}/{num_batches}...')

        # 1. Generate Z (CPU -> GPU)
        all_z = []
        for seed in batch_seeds:
            # RandomState is fast enough
            all_z.append(np.random.RandomState(seed).randn(1, G.z_dim))
        z_batch = torch.from_numpy(np.concatenate(all_z, axis=0)).to(device)
        label_batch = label_one.repeat(current_bs, 1)

        # 2. Inference
        img = G(z_batch, label_batch, truncation_psi=truncation_psi, noise_mode=noise_mode)
        
        # 3. GPU Post-processing
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255)

        # 4. Optional GPU Resize (Huge speedup for I/O)
        if resize is not None:
            # Permute back to NCHW for interpolate, then back to NHWC
            img = img.permute(0, 3, 1, 2)
            img = F.interpolate(img, size=(resize, resize), mode='area') # Area is best for downscaling
            img = img.permute(0, 2, 3, 1)

        img = img.to(torch.uint8)

        # 5. Async Save
        # Transfer to CPU
        img_cpu = img.cpu().numpy()
        
        for i, seed in enumerate(batch_seeds):
            file_path = f'{outdir}/seed{seed:04d}.png'
            # Offload PNG compression to thread pool
            executor.submit(save_image_task, img_cpu[i], file_path)

    # Wait for all saves to finish
    print("Waiting for pending writes...")
    executor.shutdown(wait=True)
    
    total_time = time.time() - start_time
    print(f"Done. Generated {len(seeds)} images in {total_time:.2f}s ({len(seeds)/total_time:.1f} imgs/sec)")

if __name__ == "__main__":
    generate_images()