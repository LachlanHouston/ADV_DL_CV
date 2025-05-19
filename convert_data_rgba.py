import torch
import h5py
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
import time
import multiprocessing
from multiprocessing import Pool

def resize_rgba_image(rgba, target_size):
    """Resize RGBA image to target size while preserving all channels."""
    # Resize each channel separately to maintain alpha
    resized = cv2.resize(rgba, target_size, interpolation=cv2.INTER_AREA)
    return resized

def process_sample(args):
    """Process a single sample for multiprocessing"""
    sample_idx, h5_path, num_layers, target_size = args
    
    processed_layers = []
    with h5py.File(h5_path, 'r') as f:
        for layer_idx in range(num_layers):
            # Get RGBA image
            rgba = f['images'][sample_idx, layer_idx]
            
            # Resize while preserving RGBA
            resized = resize_rgba_image(rgba, target_size)
            
            # Normalize to [0, 1] and convert to float32
            normalized = resized.astype(np.float32) / 255.0
            
            processed_layers.append(normalized)
    
    # Stack all layers for this sample into a single array
    return np.stack(processed_layers)

def convert_h5_to_pt(h5_path, output_dir, target_size=(64, 64), max_samples=None, num_workers=None):
    """
    Convert H5 dataset to PyTorch tensors for faster loading while preserving RGBA format
    
    Args:
        h5_path: Path to input H5 file
        output_dir: Directory to save output files
        target_size: Size to resize images to
        max_samples: Maximum number of samples to convert
        num_workers: Number of parallel workers
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine worker count
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Analyzing dataset structure in {h5_path}...")
    
    # Get dataset dimensions
    with h5py.File(h5_path, 'r') as f:
        num_samples = f['images'].shape[0]
        num_layers = f['images'].shape[1]
        original_shape = f['images'][0, 0].shape
        
        if max_samples is not None:
            num_samples = min(max_samples, num_samples)
    
    print(f"Dataset contains {num_samples} samples with {num_layers} layers each")
    print(f"Original shape: {original_shape}, Target shape: {target_size + (4,)}")  # RGBA has 4 channels
    
    # Prepare arguments for parallel processing
    args_list = [(i, h5_path, num_layers, target_size) for i in range(num_samples)]
    
    # Process samples in parallel
    print(f"Converting samples using {num_workers} workers...")
    start_time = time.time()
    
    with Pool(processes=num_workers) as pool:
        # Process all samples and show progress
        all_samples = list(tqdm(
            pool.imap(process_sample, args_list),
            total=len(args_list),
            desc="Converting samples"
        ))
    
    # Convert list of numpy arrays to a single tensor
    print("Combining samples into tensors...")
    
    # Stack all processed samples along batch dimension
    all_samples_array = np.stack(all_samples)
    
    # Convert to PyTorch tensor
    all_samples_tensor = torch.from_numpy(all_samples_array)
    
    # Get tensor properties
    print(f"Final tensor shape: {all_samples_tensor.shape}")
    
    # Save the full dataset as a single PyTorch tensor
    dataset_path = os.path.join(output_dir, "full_dataset_rgba.pt")
    print(f"Saving dataset to {dataset_path}")
    torch.save(all_samples_tensor, dataset_path)
    
    # Also save metadata
    metadata = {
        'num_samples': num_samples,
        'num_layers': num_layers,
        'target_size': target_size,
        'original_shape': original_shape,
        'tensor_shape': all_samples_tensor.shape,
        'channels': 4,  # RGBA
        'format': 'RGBA'
    }
    
    metadata_path = os.path.join(output_dir, "metadata_rgba.pt")
    torch.save(metadata, metadata_path)
    
    # Calculate timing and statistics
    total_time = time.time() - start_time
    print(f"Conversion completed in {total_time:.2f} seconds")
    print(f"Processing speed: {num_samples / total_time:.2f} samples/second")
    
    # Calculate file size
    dataset_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f"Dataset file size: {dataset_size_mb:.2f} MB")
    
    return dataset_path, metadata_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert H5 dataset to PyTorch tensors (RGBA format)")
    parser.add_argument("--input", required=True, help="Path to input H5 file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", type=int, nargs=2, default=[64, 64], help="Target size (width height)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to convert")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    convert_h5_to_pt(
        args.input,
        args.output,
        target_size=tuple(args.size),
        max_samples=args.max_samples,
        num_workers=args.workers
    )