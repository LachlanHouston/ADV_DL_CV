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
from src.utils.weighted_mse_loss import WeightedMSELoss
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss

# --- Hyperparameters ---
subset_fraction = 1.
greyscale = True
batch_size = 128
num_epochs = 20
learning_rate = 1e-3



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


# --- Model ---
class LayerGenerator(nn.Module):
    """
    A simple vision transformer based generator.
    It splits the input (composite image) into patches,
    processes them with transformer encoder blocks (conditioned on layer index),
    and decodes back to an image.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=4, embed_dim=256, num_transformer_layers=6, num_heads=8, num_layers_max=18): # Added num_layers_max
        super(LayerGenerator, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding layer (converts image to patch tokens)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learnable positional embeddings for spatial location
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # --- Add Layer Embedding ---
        self.num_layers_max = num_layers_max
        # Simple layer embedding (similar to positional embedding but for layer index)
        # Alternatively, use nn.Embedding(num_layers_max, embed_dim) and project/reshape if needed
        self.layer_embed = nn.Embedding(num_layers_max, embed_dim)
        # --- End Layer Embedding ---

        # Transformer encoder
        # Ensure batch_first=True if using standard nn.TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True) # Added batch_first=True
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Decoder: map each token back to a patch (flattened pixels)
        # The decoder input dimension might need adjustment if layer embedding changes token structure significantly.
        # This simple linear decoder assumes the output tokens from the transformer have the original embed_dim.
        # Output channels should match the target image channels (in_chans usually)
        self.decoder_out_chans = in_chans # Output channels should match target
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * self.decoder_out_chans)

        # Optional: More sophisticated decoder (e.g., convolutional)
        # Example: Transposed convolutions to upsample from transformer output
        # self.decoder_head = nn.Linear(embed_dim, some_intermediate_dim)
        # self.decoder_upsample = nn.Sequential(...) # ConvTranspose2d layers etc.

    def forward(self, x, layer_idx): # Add layer_idx as input
        # x: (B, in_chans, img_size, img_size)
        # layer_idx: (B,) tensor of layer indices
        B = x.size(0)
        in_chans_runtime = x.size(1) # Get channels from input tensor

        # Convert image to patch tokens
        x = self.patch_embed(x)  # (B, embed_dim, H', W') with H'=W'=img_size/patch_size
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # --- Incorporate Layer Embedding ---
        # Get layer embedding: (B, embed_dim)
        l_embed = self.layer_embed(layer_idx)
        # Add layer embedding to each patch token (broadcasts)
        # Unsqueeze to make it (B, 1, embed_dim) for broadcasting
        l_embed = l_embed.unsqueeze(1)
        x = x + l_embed # Add layer embedding to patch tokens
        # --- End Layer Embedding Incorporation ---

        # Add positional encoding
        x = x + self.pos_embed

        # Pass through transformer blocks
        x = self.transformer(x)  # (B, num_patches, embed_dim)

        # Decode each patch token to patch pixels
        x = self.decoder(x)  # (B, num_patches, patch_size*patch_size*decoder_out_chans)

        # Reshape to image (B, decoder_out_chans, img_size, img_size)
        # Use einops for potentially clearer reshaping
        # from einops.layers.torch import Rearrange
        # rearrange_op = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        #                         h=self.img_size // self.patch_size,
        #                         w=self.img_size // self.patch_size,
        #                         p1=self.patch_size, p2=self.patch_size)
        # x = rearrange_op(x)
        # Manual reshape:
        patches_h = self.img_size // self.patch_size
        patches_w = self.img_size // self.patch_size
        x = x.view(B, patches_h, patches_w, self.patch_size, self.patch_size, self.decoder_out_chans)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, self.decoder_out_chans, self.img_size, self.img_size)


        # Use sigmoid to restrict outputs to [0, 1]
        x = torch.sigmoid(x)
        return x


# --- Training Loop ---
def train_model(greyscale=False, subset_fraction=1.0):
    # ... (keep existing setup: data_file, img_size, transform) ...
    data_file = 'data/full_dataset.pt' # Make sure this is the correct path
    img_size = 64
    transform = None # Keep transform simple initially if using LayerPtDataset directly
    # Example transform if needed (ensure consistency with dataset)
    # transform = transforms.Compose([transforms.ToTensor()])

    # Create dataset and dataloader
    # Ensure LayerPtDataset is imported and used
    from src.data import LayerPtDataset # Make sure this import is correct
    dataset = LayerPtDataset(data_file, img_size=img_size, transform=transform,
                          greyscale=greyscale, subset_fraction=subset_fraction)
    # Consider using more workers if I/O is a bottleneck
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Added pin_memory

    # Initialize model with correct number of input channels and max layers
    in_chans = 1 if greyscale else 4 # Determine based on loaded data/greyscale flag
    # --- Get max layers from dataset if possible ---
    num_layers_max = dataset.num_layers if hasattr(dataset, 'num_layers') else 18 # Default or get from dataset
    model = LayerGenerator(img_size=img_size, patch_size=8, in_chans=in_chans,
                        embed_dim=256, num_transformer_layers=6, num_heads=8,
                        num_layers_max=num_layers_max) # Pass num_layers_max

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Consider using WeightedMSELoss as well, if changes are sparse
    # criterion = WeightedMSELoss(threshold=0.05)
    criterion = WeightedProportionalMSELoss(threshold=0.05) # Or keep this one
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    print(f"Starting training with batch size {batch_size}, epochs {num_epochs}, lr {learning_rate}")
    print(f"Dataset size: {len(dataset)}, Dataloader batches: {len(dataloader)}")


    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_start_time = time.time() # Time batches
        for i, (input_image, target, layer_idx) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")): # Unpack layer_idx
            # Move data to device
            input_image = input_image.to(device, non_blocking=True) # Use non_blocking with pin_memory
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True) # Move layer_idx to device

            optimizer.zero_grad()

            # --- Pass layer_idx to model ---
            output = model(input_image, layer_idx)

            # Check shapes
            if output.shape != target.shape:
                 print(f"Shape mismatch! Output: {output.shape}, Target: {target.shape}")
                 # Potentially resize output or debug decoder
                 # output = transforms.functional.resize(output, target.shape[-2:])

            loss = criterion(input_image, output, target) # Pass input for weighted loss calculation

            # Check for NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected at epoch {epoch+1}, batch {i}. Skipping batch.")
                # Consider saving state for debugging: torch.save(...)
                continue # Skip backprop and optimizer step for this batch


            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() # Accumulate loss directly

            # Print batch time periodically
            # if (i + 1) % 100 == 0: # Print every 100 batches
            #     batch_time = time.time() - batch_start_time
            #     print(f"  Batch {i+1}/{len(dataloader)} - Time: {batch_time:.2f}s - Current Batch Loss: {loss.item():.4f}")
            #     batch_start_time = time.time()


        # Calculate average loss for the epoch
        epoch_loss /= len(dataloader) # Average over number of batches
        print(f"Epoch {epoch+1}/{num_epochs} Average Loss: {epoch_loss:.4f}")

        # Optional: Update learning rate scheduler
        # scheduler.step()

        # Save images periodically (using the last batch of the epoch)
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
             # Ensure the tensors are on CPU for plotting
            save_image_grid(input_image.cpu(), target.cpu(), output.cpu(), epoch)


    # Save the trained model
    model_save_path = 'layer_generator_with_layer_embed.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved as {model_save_path}")

# ... (keep save_image_grid function and the if __name__ == '__main__': block) ...

if __name__ == '__main__':
    import time # Add time import if not present
    start_train_time = time.time()
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    end_train_time = time.time()
    print(f"Total training time: {end_train_time - start_train_time:.2f} seconds")