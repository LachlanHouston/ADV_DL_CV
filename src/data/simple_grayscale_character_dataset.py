import torch
from torch.utils.data import Dataset, DataLoader
import os
import time

class FastPtDataset(Dataset):
    """
    Fast dataset that loads pre-processed PyTorch tensors.
    """
    def __init__(self, dataset_path, metadata_path=None):
        """
        Initialize dataset from pre-processed PyTorch tensors
        
        Args:
            dataset_path: Path to the saved PyTorch tensor file (.pt)
            metadata_path: Path to the metadata file (optional)
        """
        print(f"Loading dataset from {dataset_path}...")
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
        print(f"Dataset loaded in {loading_time:.2f} seconds")
        print(f"Dataset shape: {self.all_data.shape}, Training pairs: {len(self.pairs)}")
        
        # Calculate memory usage
        memory_mb = self.all_data.element_size() * self.all_data.nelement() / (1024 * 1024)
        print(f"Memory usage: {memory_mb:.2f} MB")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get a training pair (input layer, target layer)"""
        sample_idx, layer_idx = self.pairs[idx]
        
        # Get input layer
        input_layer = self.all_data[sample_idx, layer_idx].unsqueeze(0)  # Add channel dimension
        
        # Get target (next) layer
        target_layer = self.all_data[sample_idx, layer_idx + 1].unsqueeze(0)  # Add channel dimension
        
        return input_layer, target_layer, layer_idx

def get_pt_dataloaders(dataset_path, metadata_path=None, batch_size=32, train_ratio=0.8):
    """
    Create train and validation dataloaders from pre-processed PyTorch tensors
    
    Args:
        dataset_path: Path to the saved PyTorch tensor file
        metadata_path: Path to the metadata file (optional)
        batch_size: Batch size for training
        train_ratio: Ratio of data to use for training
        
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Create dataset
    dataset = FastPtDataset(dataset_path, metadata_path)
    
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
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--dataset", required=True, help="Path to dataset file")
    parser.add_argument("--metadata", default=None, help="Path to metadata file")
    
    args = parser.parse_args()
    
    # Test dataset loading
    dataset = FastPtDataset(args.dataset, args.metadata)
    
    # Test data access
    input_layer, target_layer, layer_idx = dataset[0]
    print(f"Sample 0 - Input shape: {input_layer.shape}, Target shape: {target_layer.shape}")
    
    # Test dataloader
    train_loader, val_loader = get_pt_dataloaders(args.dataset, args.metadata, batch_size=4)
    
    # Get a batch
    batch = next(iter(train_loader))
    inputs, targets, layer_idxs = batch
    print(f"Batch - Inputs: {inputs.shape}, Targets: {targets.shape}, Layer indices: {layer_idxs}")