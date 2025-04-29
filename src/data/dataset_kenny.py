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

# --- New Dataset for PyTorch (.pt) files ---
class LayerPtDataset(Dataset):
    """
    Creates training pairs from a pre-processed PyTorch tensor file (.pt).
    For each sample (assumed shape: num_layers x H x W x C), each transition (layers 0..t -> layer t+1)
    becomes one training example.
    """
    def __init__(self, pt_file, img_size=64, transform=None, greyscale=False, subset_fraction=1.0):
        self.pt_file = pt_file
        self.img_size = img_size
        self.greyscale = greyscale

        # Define base transforms similar to LayerDataset
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size))
        ]

        if self.greyscale:
            base_transforms.append(transforms.Grayscale(num_output_channels=1))

        base_transforms.append(transforms.ToTensor())

        # If additional transforms are provided, add them after the base transforms
        if transform is not None:
            base_transforms.extend(transform.transforms)
        self.transform = transforms.Compose(base_transforms)

        # Load the tensor data from the .pt file
        self.all_data = torch.load(pt_file)
        self.num_samples = self.all_data.shape[0]
        self.num_layers = self.all_data.shape[1] # Assuming data is [samples, layers, H, W, C] or similar

        # Correct dimension inference if needed (e.g., if data is [samples, layers, C, H, W])
        # Example check: you might need to adjust index based on your .pt file structure
        # if self.all_data.dim() == 5:
        #    self.num_layers = self.all_data.shape[1]
        # else:
        #    # Handle other potential shapes or raise an error
        #    raise ValueError("Unexpected data shape in .pt file")


        # Build a list of (sample index, transition index) tuples
        self.indices = []
        num_samples_to_use = max(1, int(self.num_samples * subset_fraction))
        for i in range(num_samples_to_use):
            # Adjusted range to account for num_layers correctly
            for t in range(self.num_layers - 1): # We need t and t+1
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, layer_idx = self.indices[idx]
        # Retrieve the sample (adjust indices based on your data shape)
        # Assuming shape [samples, layers, H, W, C] for uint8 or [samples, layers, C, H, W] for float
        sample = self.all_data[sample_idx]

        # Handle data type and format conversion if necessary
        if isinstance(sample, torch.Tensor) and sample.dtype == torch.uint8:
             # Permute if C is last: [layers, H, W, C] -> [layers, C, H, W]
            if sample.shape[-1] == 1 or sample.shape[-1] == 3 or sample.shape[-1] == 4:
                 sample = sample.permute(0, 3, 1, 2)
            sample = sample.float() / 255.0
        elif isinstance(sample, np.ndarray) and sample.dtype == np.uint8:
            sample = torch.from_numpy(sample).float() / 255.0
             # Permute if C is last: [layers, H, W, C] -> [layers, C, H, W]
            if sample.shape[-1] == 1 or sample.shape[-1] == 3 or sample.shape[-1] == 4:
                 sample = sample.permute(0, 3, 1, 2)


        # Get the input (current layer) and target (next layer)
        # Assuming shape is now [layers, C, H, W]
        input_image = sample[layer_idx]
        target = sample[layer_idx + 1]

        # Apply transforms (which expect CHW)
        if self.transform:
            # Apply transforms; note ToPILImage expects CHW or HW
            input_image = self.transform(input_image)
            target = self.transform(target)
        # else: # If no transform, ensure output is tensor CHW
        #    input_image = torch.tensor(input_image) # Ensure tensor
        #    target = torch.tensor(target)     # Ensure tensor

        # --- Return layer_idx ---
        return input_image, target, layer_idx # Return the layer index
    
# --- New Dataset for getting the the last layer within the data --
class LastLayerPtDataset(Dataset):
    """
    Creates training pairs from a pre-processed PyTorch tensor file (.pt).
    For each sample (assumed shape: num_layers x H x W x C), each transition (layers 0..t -> layer t+1)
    becomes one training example.
    """
    def __init__(self, pt_file, img_size=64, transform=None, greyscale=False, subset_fraction=1.0):
        self.pt_file = pt_file
        self.img_size = img_size
        self.greyscale = greyscale

        # Define base transforms similar to LayerDataset
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size))
        ]

        if self.greyscale:
            base_transforms.append(transforms.Grayscale(num_output_channels=1))

        base_transforms.append(transforms.ToTensor())

        # If additional transforms are provided, add them after the base transforms
        if transform is not None:
            base_transforms.extend(transform.transforms)
        self.transform = transforms.Compose(base_transforms)

        # Load the tensor data from the .pt file
        self.all_data = torch.load(pt_file)
        self.num_samples = self.all_data.shape[0]
        self.num_layers = self.all_data.shape[1] # Assuming data is [samples, layers, H, W, C] or similar

        # Correct dimension inference if needed (e.g., if data is [samples, layers, C, H, W])
        # Example check: you might need to adjust index based on your .pt file structure
        # if self.all_data.dim() == 5:
        #    self.num_layers = self.all_data.shape[1]
        # else:
        #    # Handle other potential shapes or raise an error
        #    raise ValueError("Unexpected data shape in .pt file")


        # Build a list of (sample index, transition index) tuples
        self.indices = []
        num_samples_to_use = max(1, int(self.num_samples * subset_fraction))
        for i in range(num_samples_to_use):
            # Adjusted range to account for num_layers correctly
            for t in range(self.num_layers - 1): # We need t and t+1
                self.indices.append((i, t))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, layer_idx = self.indices[idx]
        # Retrieve the sample (adjust indices based on your data shape)
        # Assuming shape [samples, layers, H, W, C] for uint8 or [samples, layers, C, H, W] for float
        sample = self.all_data[sample_idx]

        # Handle data type and format conversion if necessary
        if isinstance(sample, torch.Tensor) and sample.dtype == torch.uint8:
             # Permute if C is last: [layers, H, W, C] -> [layers, C, H, W]
            if sample.shape[-1] == 1 or sample.shape[-1] == 3 or sample.shape[-1] == 4:
                 sample = sample.permute(0, 3, 1, 2)
            sample = sample.float() / 255.0
        elif isinstance(sample, np.ndarray) and sample.dtype == np.uint8:
            sample = torch.from_numpy(sample).float() / 255.0
             # Permute if C is last: [layers, H, W, C] -> [layers, C, H, W]
            if sample.shape[-1] == 1 or sample.shape[-1] == 3 or sample.shape[-1] == 4:
                 sample = sample.permute(0, 3, 1, 2)


        # Get the input (current layer) and target (next layer)
        # Assuming shape is now [layers, C, H, W]
        target = sample[-1]

        # Apply transforms (which expect CHW)
        if self.transform:
            # Apply transforms; note ToPILImage expects CHW or HW
            target = self.transform(target)
        # else: # If no transform, ensure output is tensor CHW
        #    input_image = torch.tensor(input_image) # Ensure tensor
        #    target = torch.tensor(target)     # Ensure tensor

        # --- Return layer_idx ---
        return target # Return the layer index
    
# Test LastLayerPtDataset
if __name__ == "__main__":
    # Example usage
    pt_file = 'data/full_dataset.pt'
    dataset = LastLayerPtDataset(pt_file, img_size=64, greyscale=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(batch.shape)  # Should print the shape of the last layer
        break