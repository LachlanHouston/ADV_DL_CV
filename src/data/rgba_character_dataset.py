import torch
from torch.utils.data import Dataset, DataLoader
import os
import time

class FastRgbaPtDataset(Dataset):
    """
    Fast dataset that loads pre-processed RGBA PyTorch tensors.
    """
    def __init__(self, dataset_path, metadata_path=None):
        """
        Initialize dataset from pre-processed PyTorch tensors
        
        Args:
            dataset_path: Path to the saved PyTorch tensor file (.pt)
            metadata_path: Path to the metadata file (optional)
        """
        print(f"Loading RGBA dataset from {dataset_path}...")
        start_time = time.time()
        
        # Load the dataset tensor directly into memory
        self.all_data = torch.load(dataset_path)
        
        # Extract dimensions
        self.num_samples, self.num_layers = self.all_data.shape[0], self.all_data.shape[1]
        
        # Load metadata if available
        if metadata_path and os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path)
            print(f"Loaded metadata: {self.metadata}")
        else:
            self.metadata = None
        
        # Create training pairs (sample_idx, layer_idx)
        self.pairs = []
        for sample_idx in range(self.num_samples):
            for layer_idx in range(self.num_layers - 1):  # -1 because we need target layer
                self.pairs.append((sample_idx, layer_idx))
        
        loading_time = time.time() - start_time
        print(f"RGBA Dataset loaded in {loading_time:.2f} seconds")
        print(f"Dataset shape: {self.all_data.shape}, Training pairs: {len(self.pairs)}")
        
        # Calculate memory usage
        memory_mb = self.all_data.element_size() * self.all_data.nelement() / (1024 * 1024)
        print(f"Memory usage: {memory_mb:.2f} MB")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a training pair (input layer, target layer)"""
        sample_idx, layer_idx = self.pairs[idx]
        
        # Get input layer - expecting RGBA data in [H, W, C] format (where C=4)
        input_layer = self.all_data[sample_idx, layer_idx]
        
        # Handle format conversion and validation
        if input_layer.dim() == 2:  # If H,W only (grayscale with no channel)
            # Add channel dim and repeat to 4 channels, then transpose to [C,H,W]
            input_layer = input_layer.unsqueeze(-1).repeat(1, 1, 4)
            input_layer = input_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        elif input_layer.dim() == 3:  # If [H,W,C] format
            if input_layer.shape[2] == 1:  # If grayscale with channel dim
                # Repeat to 4 channels and transpose
                input_layer = input_layer.repeat(1, 1, 4)
                input_layer = input_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            elif input_layer.shape[2] == 4:  # If already RGBA
                # Just transpose to PyTorch format
                input_layer = input_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            elif input_layer.shape[2] == 3:  # If RGB with no alpha
                # Add alpha channel (full opacity) and transpose
                alpha = torch.ones(input_layer.shape[0], input_layer.shape[1], 1)
                input_layer = torch.cat([input_layer, alpha], dim=2)
                input_layer = input_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            else:
                raise ValueError(f"Unexpected channel count: {input_layer.shape[2]}, expected 4 for RGBA")
        elif input_layer.dim() > 3:
            raise ValueError(f"Unexpected tensor dimensions: {input_layer.dim()}")
        
        # Get target (next) layer - similar handling
        target_layer = self.all_data[sample_idx, layer_idx + 1]
        
        # Apply the same format conversion for target
        if target_layer.dim() == 2:  # If H,W only
            target_layer = target_layer.unsqueeze(-1).repeat(1, 1, 4)
            target_layer = target_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
        elif target_layer.dim() == 3:  # If [H,W,C] format
            if target_layer.shape[2] == 1:  # If grayscale with channel dim
                target_layer = target_layer.repeat(1, 1, 4)
                target_layer = target_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            elif target_layer.shape[2] == 4:  # If already RGBA
                target_layer = target_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            elif target_layer.shape[2] == 3:  # If RGB with no alpha
                alpha = torch.ones(target_layer.shape[0], target_layer.shape[1], 1)
                target_layer = torch.cat([target_layer, alpha], dim=2)
                target_layer = target_layer.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            else:
                raise ValueError(f"Unexpected channel count: {target_layer.shape[2]}, expected 4 for RGBA")
        elif target_layer.dim() > 3:
            raise ValueError(f"Unexpected tensor dimensions: {target_layer.dim()}")
        
        return input_layer, target_layer, layer_idx

def get_rgba_pt_dataloaders(dataset_path, metadata_path=None, batch_size=32, train_ratio=0.8):
    """
    Create train and validation dataloaders from pre-processed RGBA PyTorch tensors
    
    Args:
        dataset_path: Path to the saved PyTorch tensor file
        metadata_path: Path to the metadata file (optional)
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training
        
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Create dataset
    dataset = FastRgbaPtDataset(dataset_path, metadata_path)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Simple test
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RGBA dataset loading")
    parser.add_argument("--dataset", required=True, help="Path to RGBA dataset file")
    parser.add_argument("--metadata", default=None, help="Path to metadata file")
    
    args = parser.parse_args()
    
    # Test dataset loading
    dataset = FastRgbaPtDataset(args.dataset, args.metadata)
    
    # Test data access
    input_layer, target_layer, layer_idx = dataset[0]
    print(f"Sample 0 - Input shape: {input_layer.shape}, Target shape: {target_layer.shape}")
    
    # Test dataloader
    train_loader, val_loader = get_rgba_pt_dataloaders(args.dataset, args.metadata, batch_size=4)
    
    # Get a batch
    batch = next(iter(train_loader))
    inputs, targets, layer_idxs = batch
    print(f"Batch - Inputs: {inputs.shape}, Targets: {targets.shape}, Layer indices: {layer_idxs}")