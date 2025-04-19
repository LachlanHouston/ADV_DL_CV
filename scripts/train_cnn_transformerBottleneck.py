"""
Hybrid model that combines a CNN-based U-Net with a Transformer bottleneck. 
It first applies CNN encoder blocks, feeds the compressed representation through a 
Transformer, and then reconstructs using a CNN decoder with skip connections.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LayerPtDataset # Ensure this returns layer_idx
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss

# --- Hyperparameters ---
subset_fraction = 1.0
greyscale = False
img_size = 64
patch_size = 8
batch_size = 64
num_epochs = 100
learning_rate = 1e-4
cnn_start_filters = 32
transformer_embed_dim = 256
transformer_layers = 6
transformer_heads = 8
weight_decay = 0.05

VISUALIZE_AND_CHECKPOINT_FREQUENCY = 10
SAVE_MODEL = False


# --- Configuration ---
data_file = 'data/full_dataset_color.pt'
save_dir = f'results/cnn_transformerBottleneck_{time.strftime("%Y%m%d_%H%M%S")}/'
model_save_path = os.path.join(save_dir, 'hybrid_layer_generator.pth')

def convert_for_imshow(tensor_img):
    """
    Converts a PyTorch tensor (C, H, W) to a NumPy array suitable for plt.imshow.
    Handles grayscale and swaps BGR(A) to RGB(A) if necessary for color images.
    """
    # tensor_img is shape [C, H, W]
    if tensor_img is None:
        return None, None # Handle potential None input gracefully
        
    array_img = tensor_img.cpu().detach().numpy()

    if array_img.ndim == 2: # Already grayscale H, W
        cmap = 'gray'
    elif array_img.shape[0] == 1:
        # Grayscale C, H, W -> H, W
        array_img = array_img.squeeze(0)
        cmap = 'gray'
    elif array_img.shape[0] in [3, 4]:
        # Color C, H, W -> H, W, C
        array_img = np.transpose(array_img, (1, 2, 0)) # Now shape is H, W, C

        # --- FIX: Swap BGR(A) to RGB(A) ---
        # Check number of channels AFTER transpose
        num_channels = array_img.shape[-1]
        if num_channels == 4: # Assume BGRA -> RGBA
            # Extract channels (assuming input tensor was B, G, R, A)
            b = array_img[..., 0]
            g = array_img[..., 1]
            r = array_img[..., 2]
            a = array_img[..., 3]
            # Stack in RGBA order for imshow
            array_img = np.stack([r, g, b, a], axis=-1)
        elif num_channels == 3: # Assume BGR -> RGB
            # Extract channels (assuming input tensor was B, G, R)
            b = array_img[..., 0]
            g = array_img[..., 1]
            r = array_img[..., 2]
            # Stack in RGB order for imshow
            array_img = np.stack([r, g, b], axis=-1)

        cmap = None # Let imshow handle RGB/RGBA rendering
    else: # Unexpected number of channels
        print(f"Warning: convert_for_imshow received unexpected shape {tensor_img.shape}. Displaying first channel as grayscale.")
        array_img = array_img[0] # Display first channel
        cmap = 'gray'

    # Ensure data is in a displayable range [0, 1] or [0, 255]
    # Since model output is sigmoid, data is likely [0, 1] float.
    # Clamp values just in case to avoid matplotlib warnings/errors
    if np.issubdtype(array_img.dtype, np.floating):
        array_img = np.clip(array_img, 0, 1)
    # No need to convert uint8 here as input is float tensor

    return array_img, cmap

def save_image_grid(input_images, target_images, predicted_images, epoch, save_dir='results'):
    """Save a grid of images showing input, target, and predicted results."""
    os.makedirs(save_dir, exist_ok=True)
    n_images = min(18, input_images.shape[0])
    fig, axes = plt.subplots(3, n_images, figsize=(2*n_images, 6))

    input_images = input_images.cpu().detach()
    target_images = target_images.cpu().detach()
    predicted_images = predicted_images.cpu().detach()
    input_images = input_images.float()
    target_images = target_images.float()
    predicted_images = predicted_images.float()

    for i in range(n_images):
        # Handle input
        img_input, cmap_input = convert_for_imshow(input_images[i])
        axes[0, i].imshow(img_input, cmap=cmap_input)

        # Handle target
        img_target, cmap_target = convert_for_imshow(target_images[i])
        axes[1, i].imshow(img_target, cmap=cmap_target)

        # Handle predicted
        img_pred, cmap_pred = convert_for_imshow(predicted_images[i])
        axes[2, i].imshow(img_pred, cmap=cmap_pred)

        # Titles, etc.
        if i == 0:
            axes[0, i].set_title('Input', pad=10)
            axes[1, i].set_title('Target', pad=10)
            axes[2, i].set_title('Predicted', pad=10)
        axes[0, i].axis('off')
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1:03d}.png'))
    plt.close(fig)


# --- Model Components (Corrected UpsampleBlock) ---

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BatchNorm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ****************************************************************
# Corrected UpsampleBlock Definition
# ****************************************************************
class UpsampleBlock(nn.Module):
    """
    Upsampling Block for Decoder: Upsamples input, concatenates skip features, then convolves.
    """
    def __init__(self, in_ch, skip_ch, out_ch, use_transpose_conv=True, activation=nn.GELU):
        """
        Args:
            in_ch: Channels from the previous decoder layer (or bottleneck).
            skip_ch: Channels from the corresponding encoder skip connection.
            out_ch: Output channels for this block.
            use_transpose_conv: Whether to use ConvTranspose2d or Upsample+Conv.
        """
        super().__init__()
        # Upsampling layer: Takes in_ch, outputs out_ch (or intermediate)
        if use_transpose_conv:
            # ConvTranspose output channels should match desired output *before* conv block
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1) # Adjust channels after upsampling
            )
        # Convolution layer after concatenation
        # Input channels = upsampled channels (out_ch) + skip connection channels (skip_ch)
        # Output channels = out_ch
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, activation=activation)

    def forward(self, x, skip_features):
        """
        Args:
            x: Input tensor from previous decoder layer.
            skip_features: Tensor from corresponding encoder layer.
        """
        x = self.upsample(x) # Upsample (and adjust channels if needed)
        # Concatenate along the channel dimension
        x = torch.cat([x, skip_features], dim=1)
        x = self.conv(x) # Convolve the combined features
        return x
# ****************************************************************

class LayerGeneratorHybrid(nn.Module):
    """
    Hybrid U-Net + Transformer model (Using Corrected UpsampleBlock).
    """
    def __init__(self, img_size=64, in_chans=1, out_chans=1,
                 cnn_start_filters=32, cnn_depth=4,
                 transformer_embed_dim=256, transformer_layers=6, transformer_heads=8,
                 num_layers_max=18):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_layers_max = num_layers_max
        self.cnn_depth = cnn_depth
        self.transformer_embed_dim = transformer_embed_dim

        # --- CNN Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        filters = cnn_start_filters
        encoder_channels = [] # Store output channels of each encoder block
        current_chans = in_chans
        for _ in range(cnn_depth):
            block = nn.Sequential(
                    ConvBlock(current_chans, filters),
                    ConvBlock(filters, filters)
                )
            self.encoder_blocks.append(block)
            encoder_channels.append(filters) # Store output channels of the block
            current_chans = filters
            filters *= 2
        self.encoder_channels = encoder_channels # e.g., [32, 64, 128, 256]

        # --- Bottleneck ---
        self.bottleneck_chans = current_chans # Channels before pooling at the last encoder level (e.g., 256)
        self.bottleneck_spatial_dim = img_size // (2**cnn_depth)
        self.num_patches_bottleneck = self.bottleneck_spatial_dim ** 2

        self.to_transformer_proj = nn.Conv2d(self.bottleneck_chans, transformer_embed_dim, kernel_size=1)

        # --- Transformer Bottleneck ---
        self.transformer_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_bottleneck, transformer_embed_dim))
        self.transformer_layer_embed = nn.Embedding(num_layers_max, transformer_embed_dim)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=transformer_heads,
            dim_feedforward=transformer_embed_dim * 4, dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=transformer_layers,
            norm=nn.LayerNorm(transformer_embed_dim)
        )

        # --- Projection from Transformer ---
        self.from_transformer_proj = nn.Conv2d(transformer_embed_dim, self.bottleneck_chans, kernel_size=1)

        # --- CNN Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Start decoder with bottleneck channels
        current_chans = self.bottleneck_chans # e.g., 256

        for i in range(cnn_depth):
            # Output channels for this decoder stage should match corresponding encoder stage
            out_ch = self.encoder_channels[cnn_depth - 1 - i] # e.g., 256, 128, 64, 32
            # Skip channels from the corresponding encoder stage
            skip_ch = self.encoder_channels[cnn_depth - 1 - i]
            self.decoder_blocks.append(
                UpsampleBlock(current_chans, skip_ch, out_ch)
            )
            current_chans = out_ch # Output channels become input for next stage

        # Final Convolution
        self.final_conv = nn.Conv2d(current_chans, out_chans, kernel_size=1) # current_chans should be cnn_start_filters (e.g., 32)
        self.final_act = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, layer_idx):
        B = x.shape[0]
        skip_connections = []

        # --- CNN Encoding ---
        for i in range(self.cnn_depth):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            x = self.pool(x)

        # --- Transformer Bottleneck ---
        x_proj = self.to_transformer_proj(x)
        x_flat = x_proj.flatten(2).transpose(1, 2)
        l_embed = self.transformer_layer_embed(layer_idx)
        x_tf_in = x_flat + self.transformer_pos_embed + l_embed.unsqueeze(1)
        x_tf_out = self.transformer_encoder(x_tf_in)
        x_tf_spatial = x_tf_out.transpose(1, 2).reshape(B, self.transformer_embed_dim, self.bottleneck_spatial_dim, self.bottleneck_spatial_dim)
        x = self.from_transformer_proj(x_tf_spatial) # Back to bottleneck_chans

        # --- CNN Decoding ---
        skip_connections = skip_connections[::-1]
        for i in range(self.cnn_depth):
            x = self.decoder_blocks[i](x, skip_connections[i])

        # Final output layer
        x = self.final_conv(x)
        output = self.final_act(x)

        return output

# --- Training Loop (Identical to previous version, just uses the corrected model) ---
def train_model(greyscale=True, subset_fraction=1.0):
    in_chans = 1 if greyscale else 4
    out_chans = in_chans

    print(f"Loading dataset from: {data_file}")
    try:
        dataset = LayerPtDataset(data_file, img_size=img_size, transform=None,
                              greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    print(f"Dataloader created. Batches per epoch: {len(dataloader)}")

    num_layers_max = dataset.num_layers if hasattr(dataset, 'num_layers') else 18
    # **** Instantiate the Hybrid Model ****
    model = LayerGeneratorHybrid(
        img_size=img_size, in_chans=in_chans, out_chans=out_chans,
        cnn_start_filters=cnn_start_filters, cnn_depth=4,
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_layers_max=num_layers_max
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")

    criterion = WeightedProportionalMSELoss(threshold=0.05).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(dataloader)
    # Add small epsilon to T_max to prevent division by zero if total_steps is 0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    print(f"\n--- Starting Training (Hybrid Model v3) ---")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Weight Decay: {weight_decay}")
    start_train_time = time.time()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                input_image, target, layer_idx = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch data at index {batch_idx}: {e}")
                continue

            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                output = model(input_image, layer_idx)
                if output.shape != target.shape:
                    # Try to resize output if shape mismatch is only spatial (H, W)
                    if output.shape[0:2] == target.shape[0:2] and len(output.shape) == 4 and len(target.shape) == 4:
                         print(f"\nWarning: Resizing output {output.shape} to match target {target.shape} at batch {batch_idx}.")
                         output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                         print(f"\nError: Unrecoverable shape mismatch! Output: {output.shape}, Target: {target.shape}. Skipping batch {batch_idx}.")
                         continue # Skip if shapes don't match in batch or channel dims

                loss = criterion(input_image, output, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        # --- End of Epoch ---
        if len(dataloader) > 0:
             avg_epoch_loss = epoch_loss / len(dataloader)
             current_lr = scheduler.get_last_lr()[0]
             print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_epoch_loss:.4f} - LR: {current_lr:.2e}")
        else:
             print(f"Epoch {epoch+1}/{num_epochs} - No batches processed.")

        # Save checkpoint and visualize
        if (epoch + 1) % VISUALIZE_AND_CHECKPOINT_FREQUENCY == 0 or epoch == num_epochs - 1:
            try:
                # Use last batch data for visualization
                vis_input, vis_target, vis_layer_idx = batch_data
                vis_input = vis_input.to(device)
                vis_target = vis_target.to(device)
                vis_layer_idx = vis_layer_idx.to(device)

                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    output_vis = model(vis_input, vis_layer_idx)
                save_image_grid(vis_input, vis_target, output_vis, epoch, save_dir=save_dir)
                model.train()

                if SAVE_MODEL:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
                    torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'loss': avg_epoch_loss, }, checkpoint_path)
                    print(f"Checkpoint and visualization saved for epoch {epoch+1}")
            except Exception as e:
                 print(f"\nError during visualization/checkpointing at epoch {epoch+1}: {e}")
                 model.train()

    # --- End of Training ---
    end_train_time = time.time()
    print("-" * 50)
    print(f"Training completed in {(end_train_time - start_train_time) / 60:.2f} minutes.")
    final_model_path = os.path.join(save_dir, 'hybrid_layer_generator_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved as {final_model_path}")
    print(f"Results saved in: {save_dir}")
    print("-" * 50)

if __name__ == '__main__':
    os.makedirs(save_dir, exist_ok=True)
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)