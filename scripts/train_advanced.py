import os
import sys
import time # Added time import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
# NOTE: einops is not used in this revision, but could be added back if desired
# from einops.layers.torch import Rearrange

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure the dataset class returns layer_idx
from src.data import LayerPtDataset # Make sure this points to the corrected version
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss

# --- Hyperparameters ---
subset_fraction = 1.0
greyscale = True
img_size = 64
patch_size = 8
batch_size = 128
num_epochs = 100
learning_rate = 1e-4
embed_dim = 512
num_transformer_layers = 8
num_heads = 8
weight_decay = 0.05
decoder_start_dim = embed_dim // 2 # Dimension for decoder input, e.g., 256

# --- Configuration ---
data_file = 'data/full_dataset.pt' # Path to your preprocessed .pt dataset
save_dir = f'results/train_{time.strftime("%Y%m%d_%H%M%S")}/' # Directory with timestamp
model_save_path = os.path.join(save_dir, 'vit_layer_generator_v2.pth')


def save_image_grid(input_images, target_images, predicted_images, epoch, save_dir='results'):
    """Save a grid of images showing input, target, and predicted results."""
    os.makedirs(save_dir, exist_ok=True)
    n_images = min(7, input_images.shape[0])
    fig, axes = plt.subplots(3, n_images, figsize=(2*n_images, 6))

    # Ensure tensors are on CPU and detached
    input_images = input_images.cpu().detach()
    target_images = target_images.cpu().detach()
    predicted_images = predicted_images.cpu().detach()

    for i in range(n_images):
        is_gray = input_images.shape[1] == 1
        cmap = 'gray' if is_gray else None

        # Input image
        axes[0, i].imshow(input_images[i].squeeze().numpy(), cmap=cmap)
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title('Input', pad=10)

        # Target image
        axes[1, i].imshow(target_images[i].squeeze().numpy(), cmap=cmap)
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title('Target', pad=10)

        # Predicted image
        axes[2, i].imshow(predicted_images[i].squeeze().numpy(), cmap=cmap)
        axes[2, i].axis('off')
        if i == 0: axes[2, i].set_title('Predicted', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1:03d}.png')) # Padded epoch number
    plt.close(fig) # Close the figure to free memory


# --- Model ---
class LayerGeneratorViTConvDecoder(nn.Module):
    """
    Vision Transformer based generator with a Convolutional Decoder (Corrected).
    Encodes image patches conditioned on layer index using ViT,
    then decodes using transposed convolutions.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=1, out_chans=1,
                 embed_dim=512, num_transformer_layers=8, num_heads=8,
                 decoder_start_dim=256, num_layers_max=18): # Renamed decoder_embed_dim
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_layers_max = num_layers_max
        self.decoder_start_dim = decoder_start_dim # Dimension for decoder input

        # --- Encoder ---
        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.layer_embed = nn.Embedding(num_layers_max, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=embed_dim * 4,
                                                   dropout=0.1, activation='gelu',
                                                   batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers,
                                                         norm=nn.LayerNorm(embed_dim))

        # --- Decoder ---
        self.decoder_proj = nn.Linear(embed_dim, decoder_start_dim)
        self.patches_h = img_size // patch_size
        self.patches_w = img_size // patch_size

        # Define decoder layers explicitly, ensuring channel continuity
        # Start from decoder_start_dim at patches_h x patches_w (8x8)

        # Layer 1: Upsample 8x8 -> 16x16
        # Input: decoder_start_dim = 256 channels
        # Output: decoder_start_dim // 2 = 128 channels
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(decoder_start_dim, decoder_start_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_start_dim // 2),
            nn.GELU()
        )
        # Current channels = 128

        # Layer 2: Upsample 16x16 -> 32x32
        # Input: 128 channels
        # Output: 128 // 2 = 64 channels
        self.decoder_layer2 = nn.Sequential(
             nn.ConvTranspose2d(decoder_start_dim // 2, decoder_start_dim // 4, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm2d(decoder_start_dim // 4),
             nn.GELU()
        )
        # Current channels = 64

        # Layer 3: Upsample 32x32 -> 64x64
        # Input: 64 channels
        # Output: 64 // 2 = 32 channels
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(decoder_start_dim // 4, decoder_start_dim // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(decoder_start_dim // 8),
            nn.GELU()
        )
        # Current channels = 32

        # Final layer: 64x64 -> 64x64
        # Input: 32 channels
        # Output: out_chans (e.g., 1 for greyscale)
        self.decoder_output = nn.Sequential(
            nn.Conv2d(decoder_start_dim // 8, out_chans, kernel_size=3, padding=1),
            nn.Sigmoid() # Output activation
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
             # Use kaiming_normal_ for layers followed by ReLU/GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, layer_idx):
        B = x.shape[0]

        # --- Encoding ---
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        l_embed = self.layer_embed(layer_idx)
        x = x + l_embed.unsqueeze(1)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)

        # --- Decoding ---
        x = self.decoder_proj(x)
        x = x.transpose(1, 2).reshape(B, self.decoder_start_dim, self.patches_h, self.patches_w)

        # Apply decoder layers sequentially
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = self.decoder_layer3(x)
        x = self.decoder_output(x)

        return x

# --- Training Loop ---
def train_model(greyscale=True, subset_fraction=1.0):
    in_chans = 1 if greyscale else 4
    out_chans = in_chans

    # --- Dataset and Dataloader ---
    print(f"Loading dataset from: {data_file}")
    try:
        dataset = LayerPtDataset(data_file, img_size=img_size, transform=None,
                              greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return # Exit if dataset cannot be loaded

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataloader created. Batches per epoch: {len(dataloader)}")

    # --- Model Initialization ---
    num_layers_max = dataset.num_layers if hasattr(dataset, 'num_layers') else 18
    model = LayerGeneratorViTConvDecoder(
        img_size=img_size, patch_size=patch_size,
        in_chans=in_chans, out_chans=out_chans,
        embed_dim=embed_dim, num_transformer_layers=num_transformer_layers,
        num_heads=num_heads, decoder_start_dim=decoder_start_dim,
        num_layers_max=num_layers_max
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")

    # --- Loss and Optimizer ---
    criterion = WeightedProportionalMSELoss(threshold=0.05).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Scheduler T_max should be total steps, not epochs
    total_steps = num_epochs * len(dataloader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    # --- Training ---
    print(f"\n--- Starting Training ---")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    start_train_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Ensure batch data unpacking is correct based on LayerPtDataset output
                input_image, target, layer_idx = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch data at index {batch_idx}: {e}")
                print(f"Check the __getitem__ method of LayerPtDataset.")
                continue # Skip this batch

            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                output = model(input_image, layer_idx)
                # Ensure output shape matches target before loss calculation
                if output.shape != target.shape:
                    print(f"\nWarning: Shape mismatch! Output: {output.shape}, Target: {target.shape}. Skipping batch {batch_idx}.")
                    continue # Skip if shapes don't match

                loss = criterion(input_image, output, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping.")
                continue

            scaler.scale(loss).backward()
            # Optional: Gradient clipping (uncomment if needed)
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step scheduler each iteration

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        # --- End of Epoch ---
        if len(dataloader) > 0: # Avoid division by zero if dataloader is empty
             avg_epoch_loss = epoch_loss / len(dataloader)
             current_lr = scheduler.get_last_lr()[0]
             print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f} - LR: {current_lr:.2e}")
        else:
             print(f"Epoch {epoch+1}/{num_epochs} - No batches processed.")


        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            try:
                # Fetch last batch data again for visualization if progress_bar cleared it
                vis_input, vis_target, vis_layer_idx = batch_data
                vis_input = vis_input.to(device)
                vis_target = vis_target.to(device)
                vis_layer_idx = vis_layer_idx.to(device)

                # Visualize results periodically (using last batch)
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    output_vis = model(vis_input, vis_layer_idx)
                save_image_grid(vis_input, vis_target, output_vis, epoch, save_dir=save_dir)
                model.train()

                # Save Checkpoint
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    # Optionally save scaler state for exact resumption with AMP
                    # 'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint and visualization saved for epoch {epoch+1}")

            except Exception as e:
                 print(f"\nError during visualization/checkpointing at epoch {epoch+1}: {e}")
                 model.train() # Ensure model is back in train mode


    # --- End of Training ---
    end_train_time = time.time()
    print("-" * 50)
    print(f"Training completed in {(end_train_time - start_train_time) / 60:.2f} minutes.")
    # Save final model
    final_model_path = os.path.join(save_dir, 'vit_layer_generator_final.pth') # Use final path
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")
    print(f"Results saved in: {save_dir}")
    print("-" * 50)


if __name__ == '__main__':
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Start training
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)