import os
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LayerDataset, LayerPtDataset

def save_image_grid(input_images, target_images, predicted_images, epoch, save_dir='results'):
    """Save a grid of images showing input, target, and predicted results."""
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Take the first 7 images
    n_images = min(7, input_images.shape[0])
    
    # Create a figure with 3 rows (input, target, predicted) and n_images columns
    fig, axes = plt.subplots(3, n_images, figsize=(2*n_images, 6))
    
    for i in range(n_images):
        # Input image
        axes[0, i].imshow(input_images[i].cpu().squeeze().numpy(), cmap='gray' if input_images.shape[1] == 1 else None)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Input', pad=10)
            
        # Target image
        axes[1, i].imshow(target_images[i].cpu().squeeze().numpy(), cmap='gray' if target_images.shape[1] == 1 else None)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Target', pad=10)
            
        # Predicted image
        axes[2, i].imshow(predicted_images[i].cpu().squeeze().detach().numpy(), cmap='gray' if predicted_images.shape[1] == 1 else None)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Predicted', pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1}.png'))
    plt.close()

subset_fraction = 0.01
greyscale = True
batch_size = 5
num_epochs = 20
learning_rate = 1e-3


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
def train_model(greyscale=False, subset_fraction=1.0):
    # Hyperparameters and paths
    # data_file = 'data/kenney_dataset_1000.h5'
    data_file = 'data/full_dataset.pt'
    img_size = 64          # adjust to your image resolution if needed
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    dataset = LayerPtDataset(data_file, img_size=img_size, transform=transform, 
                          greyscale=greyscale, subset_fraction=subset_fraction)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model with correct number of input channels
    in_chans = 1 if greyscale else 4
    model = LayerGenerator(img_size=img_size, patch_size=8, in_chans=in_chans,
                        embed_dim=256, num_transformer_layers=6, num_heads=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
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

        # Save images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_image_grid(input_image, target, output, epoch)

    # Save the trained model
    torch.save(model.state_dict(), 'layer_generator.pth')
    print("Model saved as layer_generator.pth")

if __name__ == '__main__':
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)