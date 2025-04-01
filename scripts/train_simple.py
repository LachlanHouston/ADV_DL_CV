import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

# --- Dataset ---
class LayerDataset(Dataset):
    """
    Creates training pairs from the HDF5 dataset.
    For each sample (shape: num_layers x H x W x 4), each transition (layers 0..t -> layer t+1)
    becomes one training example.
    """
    def __init__(self, h5_file, dataset_name='images', transform=None):
        self.h5_file = h5_file
        self.dataset_name = dataset_name
        self.transform = transform
        with h5py.File(h5_file, 'r') as f:
            dataset = f[dataset_name]
            self.num_samples = dataset.shape[0]
            self.num_layers = dataset.shape[1]
        self.hf = None
        # Build a list of (sample index, transition index) tuples.
        self.indices = []
        for i in range(self.num_samples):
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
        target = sample[layer_idx + 1]         # Target: next layer.
        # If a transform is provided, assume it expects images in HWC format (e.g. for ToPILImage).
        # Otherwise, convert from HWC to CHW and wrap with torch.tensor.
        if self.transform:
            # Do NOT transpose; leave the image as HWC so transforms.ToPILImage can handle it.
            input_image = self.transform(input_image)
            target = self.transform(target)
        else:
            # Change from HWC to CHW format.
            input_image = np.transpose(input_image, (2, 0, 1))
            target = np.transpose(target, (2, 0, 1))
            input_image = torch.tensor(input_image, dtype=torch.float32)
            target = torch.tensor(target, dtype=torch.float32)
        return input_image, target

# --- Model ---
class LayerGenerator(nn.Module):
    """
    A simple vision transformer based generator.
    It splits the input (composite image) into patches,
    processes them with transformer encoder blocks, and decodes back to an image.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=4, embed_dim=256, num_transformer_layers=6, num_heads=8):
        super(LayerGenerator, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        # Patch embedding layer (converts image to patch tokens)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Decoder: map each token back to a patch (flattened pixels)
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * in_chans)
    
    def forward(self, x):
        # x: (B, in_chans, img_size, img_size)
        B = x.size(0)
        in_chans = x.size(1)
        # Convert image to patch tokens
        x = self.patch_embed(x)  # (B, embed_dim, H', W') with H' = W' = img_size/patch_size
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed  # Add positional encoding
        # Pass through transformer blocks
        x = self.transformer(x)  # (B, num_patches, embed_dim)
        # Decode each patch token to patch pixels
        x = self.decoder(x)  # (B, num_patches, patch_size*patch_size*in_chans)
        x = x.transpose(1, 2)
        # Reshape to image (B, in_chans, img_size, img_size)
        x = x.view(B, in_chans, self.img_size, self.img_size)
        # Use sigmoid to restrict outputs to [0, 1]
        x = torch.sigmoid(x)
        return x

# --- Training Loop ---
def train_model():
    # Hyperparameters and paths
    h5_file = 'data/kenney_dataset_1000.h5'
    dataset_name = 'images'
    img_size = 64          # adjust to your image resolution if needed
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    batch_size = 16
    num_epochs = 20
    learning_rate = 1e-3

    # Create dataset and dataloader
    dataset = LayerDataset(h5_file, dataset_name, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss, and optimizer.
    model = LayerGenerator(img_size=img_size, patch_size=8, in_chans=4,
                           embed_dim=256, num_transformer_layers=6, num_heads=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_image, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_image = input_image.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(input_image)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * input_image.size(0)
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'layer_generator.pth')
    print("Model saved as layer_generator.pth")

if __name__ == '__main__':
    train_model()