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

def rgba_to_grayscale(rgba):
    """Convert RGBA image (HxWx4) to grayscale (HxW), respecting alpha."""
    # Extract RGB and alpha
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3] # Shape (H, W)

    # Convert RGB to grayscale using OpenCV's standard weights
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) # Shape (H, W)

    # Apply alpha mask (pixels where alpha is 0 become 0)
    # Ensure alpha is in [0, 1] for multiplication
    alpha_normalized = alpha / 255.0
    gray = gray * alpha_normalized

    return gray # Shape (H, W), dtype usually float64 after multiplication

def resize_image(image, target_size):
    """Resize image (single or multi-channel) to target size (width, height)."""
    # cv2.resize expects target size as (width, height)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_sample(args):
    """Process a single sample for multiprocessing"""
    sample_idx, h5_path, num_layers, target_size, process_in_color = args

    processed_layers = []
    try: # Add try-except for better error handling in workers
        with h5py.File(h5_path, 'r') as f:
            sample_data = f['images'][sample_idx] # Load once if possible (check memory)
            for layer_idx in range(num_layers):
                # Get original RGBA image (HxWx4, uint8)
                rgba = sample_data[layer_idx] # f['images'][sample_idx, layer_idx]

                if process_in_color:
                    # --- Process in Color (RGBA) ---
                    # Resize RGBA image
                    resized_rgba = resize_image(rgba, target_size) # Shape (H, W, 4)

                    # Normalize to [0, 1] and convert to float32
                    normalized_rgba = resized_rgba.astype(np.float32) / 255.0

                    # Transpose from HWC (Height, Width, Channel) to CHW (Channel, Height, Width) for PyTorch convention
                    processed_image = np.transpose(normalized_rgba, (2, 0, 1)) # Shape (4, H, W)
                else:
                    # --- Process in Grayscale ---
                    # Convert RGBA to grayscale
                    gray = rgba_to_grayscale(rgba) # Shape (H, W)

                    # Resize grayscale image
                    resized_gray = resize_image(gray, target_size) # Shape (H, W)

                    # Normalize to [0, 1] and convert to float32
                    # Note: Grayscale conversion output might be float64, ensure final is float32
                    normalized_gray = resized_gray.astype(np.float32) / 255.0 # Max value is already handled by conversion/alpha

                    # Add channel dimension: (H, W) -> (1, H, W) for consistent stacking
                    processed_image = np.expand_dims(normalized_gray, axis=0) # Shape (1, H, W)

                processed_layers.append(processed_image)

        # Stack all layers for this sample into a single array (NumLayers, Channels, Height, Width)
        return np.stack(processed_layers)
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
        # Return None or an empty array to indicate failure, needs handling later
        # For simplicity, we'll raise it here, but in production you might want robust handling
        raise e


def convert_h5_to_pt(h5_path, output_dir, target_size=(64, 64), use_color=False, max_samples=None, num_workers=None):
    """
    Convert H5 dataset to PyTorch tensors for faster loading.

    Args:
        h5_path: Path to input H5 file
        output_dir: Directory to save output files
        target_size: Target size (width, height) to resize images to
        use_color: If True, process images in RGBA (4 channels). If False, convert to grayscale (1 channel).
        max_samples: Maximum number of samples to convert
        num_workers: Number of parallel workers
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine worker count
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2) # Use half cores as default

    print(f"Starting H5 to PT conversion...")
    print(f"  Input H5: {h5_path}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Target Size: {target_size}")
    print(f"  Color Mode: {'RGBA (4 Channels)' if use_color else 'Grayscale (1 Channel)'}")

    print(f"\nAnalyzing dataset structure in {h5_path}...")

    # Get dataset dimensions
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'images' not in f:
                 raise KeyError("Dataset 'images' not found in HDF5 file.")
            dset = f['images']
            num_total_samples = dset.shape[0]
            num_layers = dset.shape[1]
            # Assuming shape is (samples, layers, height, width, channels)
            original_shape = dset.shape[2:] # Should be H, W, C
            print(f"  Found {num_total_samples} total samples.")
            print(f"  Found {num_layers} layers per sample.")
            print(f"  Original image shape (H, W, C): {original_shape}")

            if dset.ndim != 5 or original_shape[-1] != 4:
                print(f"Warning: Expected dataset shape (samples, layers, H, W, 4), but got {dset.shape}. Proceeding, but check source data.")

    except Exception as e:
        print(f"Error accessing HDF5 file structure: {e}")
        return None, None

    num_samples_to_process = num_total_samples
    if max_samples is not None:
        num_samples_to_process = min(max_samples, num_total_samples)
        print(f"  Processing a subset of {num_samples_to_process} samples.")
    else:
        print(f"  Processing all {num_samples_to_process} samples.")


    # Prepare arguments for parallel processing
    # Pass use_color flag to each worker
    args_list = [(i, h5_path, num_layers, target_size, use_color) for i in range(num_samples_to_process)]

    # Process samples in parallel
    print(f"\nConverting samples using {num_workers} workers...")
    start_time = time.time()

    all_samples_list = []
    try:
        with Pool(processes=num_workers) as pool:
            # Process all samples and show progress
            all_samples_list = list(tqdm(
                pool.imap(process_sample, args_list, chunksize=max(1, num_samples_to_process // (num_workers * 4))), # Adjust chunksize
                total=len(args_list),
                desc="Converting samples"
            ))
        # Check if any worker returned None due to error (if implemented that way)
        if any(s is None for s in all_samples_list):
             print("\nWarning: Some samples failed during processing.")
             all_samples_list = [s for s in all_samples_list if s is not None] # Filter out failed ones
             if not all_samples_list:
                  print("Error: No samples processed successfully.")
                  return None, None

    except Exception as e:
         print(f"\nError during parallel processing: {e}")
         return None, None


    # Convert list of numpy arrays to a single tensor
    print("\nCombining samples into a single tensor...")
    try:
        # Stack all processed samples along a new batch dimension
        # Input list contains arrays of shape (NumLayers, Channels, Height, Width)
        # Output array shape: (NumSamples, NumLayers, Channels, Height, Width)
        all_samples_array = np.stack(all_samples_list)

        # Convert to PyTorch tensor
        all_samples_tensor = torch.from_numpy(all_samples_array) # Inherits float32 type
    except Exception as e:
        print(f"Error during final stacking or tensor conversion: {e}")
        # This might happen if processed layers have inconsistent shapes
        print("Check worker output or individual processed layer shapes.")
        return None, None

    # Get tensor properties
    final_shape = all_samples_tensor.shape
    print(f"  Final tensor shape (N, L, C, H, W): {final_shape}")
    expected_channels = 4 if use_color else 1
    if final_shape[2] != expected_channels:
         print(f"Warning: Expected {expected_channels} channels based on 'use_color' flag, but tensor has {final_shape[2]} channels.")

    # Save the full dataset as a single PyTorch tensor
    dataset_filename = "full_dataset_256.pt" if use_color else "full_dataset_gray.pt"
    dataset_path = os.path.join(output_dir, dataset_filename)
    print(f"Saving dataset to {dataset_path}...")
    try:
        torch.save(all_samples_tensor, dataset_path)
    except Exception as e:
        print(f"Error saving tensor to {dataset_path}: {e}")
        return None, None

    # Also save metadata
    metadata = {
        'num_samples': final_shape[0], # Use actual processed count
        'num_layers': final_shape[1],
        'num_channels': final_shape[2],
        'height': final_shape[3],
        'width': final_shape[4],
        'color_mode': use_color,
        'target_size': target_size,
        'source_h5': os.path.basename(h5_path),
        'original_image_shape': original_shape, # From HDF5 file analysis
        'tensor_shape': final_shape,
    }

    metadata_filename = "metadata_color.pt" if use_color else "metadata_gray.pt"
    metadata_path = os.path.join(output_dir, metadata_filename)
    print(f"Saving metadata to {metadata_path}...")
    try:
         torch.save(metadata, metadata_path)
    except Exception as e:
         print(f"Error saving metadata to {metadata_path}: {e}")
         # Continue, dataset is saved, but metadata failed

    # Calculate timing and statistics
    total_time = time.time() - start_time
    print(f"\nConversion completed in {total_time:.2f} seconds")
    if total_time > 0:
        print(f"  Processing speed: {final_shape[0] / total_time:.2f} samples/second")

    # Calculate file size
    try:
        dataset_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"  Dataset file size: {dataset_size_mb:.2f} MB")
    except OSError:
        print("  Could not retrieve dataset file size.") # Might happen if save failed silently

    print("-" * 50)

    return dataset_path, metadata_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert H5 dataset containing RGBA image sequences to PyTorch tensors (.pt)")
    parser.add_argument("--input", default='data/kenney_dataset_1000.h5', help="Path to input H5 file (expected structure: dset['images'] with shape [samples, layers, H, W, 4])")
    parser.add_argument("--output", default='data/', help="Output directory for .pt files")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Target size (width height), e.g., --size 64 64")
    # Add the color argument as a flag
    parser.add_argument("--color", action='store_true', default=True, help="Process and save images in color (RGBA, 4 channels).")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to convert (optional)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (optional, defaults to CPU count / 2)")

    args = parser.parse_args()

    convert_h5_to_pt(
        h5_path=args.input,
        output_dir=args.output,
        target_size=tuple(args.size),
        use_color=args.color, # Pass the flag value
        max_samples=args.max_samples,
        num_workers=args.workers
    )