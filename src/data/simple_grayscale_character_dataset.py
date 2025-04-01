import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import cv2

class GrayscaleLayerDataset(Dataset):
    """
    Dataset for loading character layers from HDF5 and converting to grayscale.
    """
    def __init__(self, h5_path, target_size=(64, 64), max_samples=None):
        """
        Args:
            h5_path: Path to the HDF5 file
            target_size: Size to resize images to (default: 64x64)
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.h5_path = h5_path
        self.target_size = target_size
        
        # Open the HDF5 file to get metadata
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = f['images'].shape[0]
            self.num_layers = f['images'].shape[1]
            
            if max_samples is not None:
                self.num_samples = min(max_samples, self.num_samples)
        
        # Create all possible (sample_idx, input_layer_idx) pairs for training
        self.pairs = []
        for sample_idx in range(self.num_samples):
            for layer_idx in range(self.num_layers - 1):  # -1 because we need a target
                self.pairs.append((sample_idx, layer_idx))
    
    def __len__(self):
        return len(self.pairs)
    
    def rgba_to_grayscale(self, rgba):
        """Convert RGBA image to grayscale, respecting alpha channel."""
        # Extract RGB and alpha
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4]
        
        # Convert RGB to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply alpha mask (this preserves transparency)
        gray = gray * (alpha[:, :, 0] / 255.0)
        
        return gray
    
    def resize_image(self, image):
        """Resize image to target size."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def __getitem__(self, idx):
        sample_idx, layer_idx = self.pairs[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get input and target layers (RGBA format)
            input_rgba = f['images'][sample_idx, layer_idx]
            target_rgba = f['images'][sample_idx, layer_idx + 1]
            
            # Convert to grayscale and resize
            input_gray = self.rgba_to_grayscale(input_rgba)
            target_gray = self.rgba_to_grayscale(target_rgba)
            
            input_resized = self.resize_image(input_gray)
            target_resized = self.resize_image(target_gray)
            
            # Convert to torch tensors and normalize to [0, 1]
            input_tensor = torch.from_numpy(input_resized).float() / 255.0
            target_tensor = torch.from_numpy(target_resized).float() / 255.0
            
            # Add channel dimension
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)
            
            return input_tensor, target_tensor, layer_idx

def get_grayscale_dataloaders(h5_path, target_size=(64, 64), batch_size=32, 
                             train_ratio=0.8, max_samples=None):
    """
    Create train and validation dataloaders for grayscale character generation.
    
    Args:
        h5_path: Path to the HDF5 file
        target_size: Size to resize images to (default: 64x64)
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        max_samples: Maximum number of samples to use (for debugging)
        
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Create dataset
    dataset = GrayscaleLayerDataset(h5_path, target_size, max_samples)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader


if __name__ == "__main__":

    train_dataloader, val_dataloader = get_grayscale_dataloaders("data/kenney_dataset_10.h5")

    print("")