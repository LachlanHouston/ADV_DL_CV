import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- Dataset ---
class LayerDataset(Dataset):
    """
    Creates training pairs from the HDF5 dataset.
    For each sample (shape: num_layers x H x W x 4), each transition (layers 0..t -> layer t+1)
    becomes one training example.
    """
    def __init__(self, h5_file, dataset_name='images', img_size=64, transform=None, greyscale=False, subset_fraction=1.0):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.greyscale = greyscale
        
        # Define transforms based on greyscale option
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size))
        ]
        
        if self.greyscale:
            base_transforms.append(transforms.Grayscale(num_output_channels=1))
            
        base_transforms.append(transforms.ToTensor())
        
        # If additional transforms are provided, add them after our base transforms
        if transform is not None:
            base_transforms.extend(transform.transforms)
            
        self.transform = transforms.Compose(base_transforms)
            
        with h5py.File(h5_file, 'r') as f:
            dataset = f[dataset_name]
            self.num_samples = dataset.shape[0]
            self.num_layers = dataset.shape[1]
            
        self.hf = None
        # Build a list of (sample index, transition index) tuples.
        self.indices = []
        # Calculate the number of samples to use based on the fraction
        num_samples_to_use = max(1, int(self.num_samples * subset_fraction))
        
        for i in range(num_samples_to_use):
            for t in range(self.num_layers - 1):
                self.indices.append((i, t))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sample_idx, layer_idx = self.indices[idx]
        # Check to open the file if it hasn't been opened in the current worker process
        if self.hf is None:
            self.hf = h5py.File(self.h5_file, 'r')
        # Load the full sample (shape: num_layers, H, W, 4)
        sample = self.hf[self.dataset_name][sample_idx]
        # Convert from uint8 to float in [0, 1]
        sample = sample.astype(np.float32) / 255.0
        # Since layers are accumulative, the current layer already includes previous details.
        input_image = sample[layer_idx]      # Input: current accumulative layer.
        target = sample[layer_idx + 1]       # Target: next layer.
        # If a transform is provided, assume it expects images in HWC format (e.g. for ToPILImage).
        # Otherwise, convert from HWC to CHW and wrap with torch.tensor.
        if self.transform:
            input_image = self.transform(input_image)
            target = self.transform(target)
        else:
            input_image = np.transpose(input_image, (2, 0, 1))
            target = np.transpose(target, (2, 0, 1))
            input_image = torch.tensor(input_image, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
        return input_image, target