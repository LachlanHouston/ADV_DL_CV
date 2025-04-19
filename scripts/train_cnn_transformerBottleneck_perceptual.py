"""
Hybrid model that combines a CNN-based U-Net with a Transformer bottleneck.
It first applies CNN encoder blocks, feeds the compressed representation through a
Transformer, and then reconstructs using a CNN decoder with skip connections.

This version incorporates VGG Perceptual Loss alongside the original
WeightedProportionalMSELoss to improve fine detail generation.
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
import torchvision.models as models # Added for VGG
from torchvision.transforms.functional import normalize # Added for VGG normalization
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LayerPtDataset # Ensure this returns layer_idx
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss

# --- Hyperparameters ---
subset_fraction = 1.0     # Use 1.0 for full dataset, smaller for testing
greyscale = False         # Set to True for grayscale, False for color (4 channels assumed)
img_size = 64             # Input/Output image size (consider increasing to 128 or 256 for details)
patch_size = 8            # Relevant if using patch embedding elsewhere, not directly here
batch_size = 32           # Adjust based on GPU memory (might need to decrease)
num_epochs = 100          # Number of training epochs
learning_rate = 1e-4      # Initial learning rate
cnn_start_filters = 32    # Number of filters in the first CNN layer
transformer_embed_dim = 256 # Embedding dimension in the Transformer bottleneck
cnn_depth = 4
transformer_layers = 6    # Number of layers in the Transformer bottleneck
transformer_heads = 8     # Number of attention heads in the Transformer
weight_decay = 0.05       # Weight decay for AdamW optimizer

# Loss Function Weights (IMPORTANT: TUNE THESE)
lambda_mse = 1.0          # Weight for WeightedProportionalMSELoss
lambda_perceptual = 0.005   # Weight for VGGPerceptualLoss (start low, e.g., 0.01-0.1)

VISUALIZE_AND_CHECKPOINT_FREQUENCY = 1     # Save images and model every N epochs
SAVE_MODEL = False                          # Set to True to save checkpoints and final model


# --- Configuration ---
data_file = 'data/full_dataset_color.pt' # Path to your dataset file
# Create a unique directory for each run
save_dir = f'results/hybrid_percLoss_{time.strftime("%Y%m%d_%H%M%S")}/'
model_save_path = os.path.join(save_dir, 'hybrid_layer_generator.pth') # Final model path (legacy, now saved in loop)


# --- VGG Perceptual Loss ---
class VGGPerceptualLoss(nn.Module):
    """
    Calculates perceptual loss using features from a pre-trained VGG19 network.
    Compares features between the generated image and the target image.
    """
    def __init__(self, feature_layers=[2, 7, 16, 25, 34], weights=[0.2, 0.2, 0.2, 0.2, 0.2]):
        """
        Args:
            feature_layers (list): Indices of VGG19 layers (often post-ReLU) to use for feature comparison.
                                   Default corresponds approx to relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 outputs.
                                   Indices refer to the sequential modules within vgg19.features.
            weights (list): Weights for the L1 loss at each feature layer. Must match len(feature_layers).
        """
        super().__init__()
        if len(feature_layers) != len(weights):
            raise ValueError("Length of feature_layers and weights must be the same.")

        self.weights = weights
        # Load VGG19 model pre-trained on ImageNet
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval() # Set to evaluation mode

        # Freeze VGG parameters - we don't train VGG
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_layers = sorted(feature_layers)
        self.max_layer_idx = max(self.feature_layers)

        # Create a sequential module containing VGG layers up to the max needed index
        self.vgg_features = nn.Sequential()
        for i, layer in enumerate(vgg.children()):
            self.vgg_features.add_module(str(i), layer)
            if i == self.max_layer_idx:
                break # Stop adding layers once we reach the last one needed

        # --- VGG Input Normalization ---
        # VGG expects inputs normalized with ImageNet mean and std
        # Register buffers so they are moved to the correct device with the module
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Use L1 loss for comparing features (common choice for perceptual loss)
        self.criterion = nn.L1Loss()

    def _normalize_for_vgg(self, image_batch):
        """ Normalizes a batch of images for VGG input. Handles grayscale/RGBA conversion. """
        num_channels = image_batch.shape[1]
        if num_channels == 1:
            # Repeat grayscale channel 3 times
            image_batch = image_batch.repeat(1, 3, 1, 1)
        elif num_channels == 4:
             # Use only RGB channels if RGBA (assuming alpha is last)
             image_batch = image_batch[:, :3, :, :]
        # Else assume it's already 3 channels (RGB or BGR)

        # Ensure input is in [0, 1] range before normalization
        image_batch = torch.clamp(image_batch, 0.0, 1.0)
        # Apply ImageNet normalization
        return (image_batch - self.mean) / self.std # Buffers are automatically on the correct device


    def forward(self, generated, target):
        """
        Calculates the perceptual loss between generated and target images.

        Args:
            generated (torch.Tensor): Batch of generated images [B, C, H, W], values in [0, 1].
            target (torch.Tensor): Batch of target images [B, C, H, W], values in [0, 1].

        Returns:
            torch.Tensor: The calculated perceptual loss (scalar).
        """
        # Normalize images for VGG
        generated_norm = self._normalize_for_vgg(generated)
        target_norm = self._normalize_for_vgg(target)

        # --- Extract features layer by layer ---
        feature_loss = 0.0
        current_gen = generated_norm
        current_target = target_norm
        last_layer_extracted = -1

        for i, layer in enumerate(self.vgg_features.children()):
            current_gen = layer(current_gen)
            current_target = layer(current_target)

            if i in self.feature_layers:
                # Find the weight corresponding to this layer index
                layer_weight = self.weights[self.feature_layers.index(i)]
                # Calculate L1 loss between features at this layer
                feature_loss += layer_weight * self.criterion(current_gen, current_target)
                last_layer_extracted = i

            # Optimization: stop iterating if we've extracted the last required feature layer
            if last_layer_extracted == self.max_layer_idx:
                break

        # Handle case where not all specified layers were found (shouldn't happen with correct indices)
        if last_layer_extracted != self.max_layer_idx:
             print(f"Warning: Max feature layer index {self.max_layer_idx} was not reached. Last extracted: {last_layer_extracted}")


        return feature_loss



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
    n_images = min(8, input_images.shape[0]) # Show fewer images to keep grid manageable
    fig, axes = plt.subplots(3, n_images, figsize=(2 * n_images, 6)) # Rows: Input, Target, Predicted

    # Ensure tensors are on CPU and detached
    input_images = input_images.cpu().detach().float()
    target_images = target_images.cpu().detach().float()
    predicted_images = predicted_images.cpu().detach().float()

    for i in range(n_images):
        # Input Image
        img_input, cmap_input = convert_for_imshow(input_images[i])
        ax = axes[0, i]
        ax.imshow(img_input, cmap=cmap_input)
        ax.axis('off')
        if i == 0: ax.set_title('Input', pad=10)

        # Target Image
        img_target, cmap_target = convert_for_imshow(target_images[i])
        ax = axes[1, i]
        ax.imshow(img_target, cmap=cmap_target)
        ax.axis('off')
        if i == 0: ax.set_title('Target', pad=10)

        # Predicted Image
        img_pred, cmap_pred = convert_for_imshow(predicted_images[i])
        ax = axes[2, i]
        ax.imshow(img_pred, cmap=cmap_pred)
        ax.axis('off')
        if i == 0: ax.set_title('Predicted', pad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1:03d}.png'))
    plt.close(fig) # Close the figure to free memory


# --- Model Components ---

class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv -> BatchNorm -> Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

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
            use_transpose_conv: Whether to use ConvTranspose2d or Upsample+Conv. Default: True
        """
        super().__init__()
        # Upsampling layer: Takes in_ch, outputs 'out_ch' channels to match skip connection before conv
        if use_transpose_conv:
            # ConvTranspose output channels should match the target 'out_ch' for this block
            self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            # Upsample first, then adjust channels with a 1x1 conv
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1) # Adjust channels to 'out_ch'
            )
        # Convolution layer after concatenation
        # Input channels = upsampled channels (now out_ch) + skip connection channels (skip_ch)
        # Output channels = desired out_ch for this block
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, activation=activation)

    def forward(self, x, skip_features):
        """
        Args:
            x: Input tensor from previous decoder layer.
            skip_features: Tensor from corresponding encoder layer.
        """
        x = self.upsample(x) # Upsample and potentially adjust channels to out_ch
        # Concatenate along the channel dimension
        x = torch.cat([x, skip_features], dim=1) # Shape: [B, out_ch + skip_ch, H, W]
        x = self.conv(x) # Convolve the combined features -> [B, out_ch, H, W]
        return x

class LayerGeneratorHybrid(nn.Module):
    """
    Hybrid U-Net + Transformer model. Includes Encoder, Transformer Bottleneck, Decoder.
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
        encoder_channels = [] # Store output channels of each encoder block for skip connections
        current_chans = in_chans
        for i in range(cnn_depth):
            block = nn.Sequential(
                ConvBlock(current_chans, filters),
                ConvBlock(filters, filters) # Keep same filter count within the block
            )
            self.encoder_blocks.append(block)
            encoder_channels.append(filters) # Store the output channels of this block
            current_chans = filters
            if i < cnn_depth -1: # Don't double filters before bottleneck projection
                 filters *= 2
        self.encoder_channels = encoder_channels # e.g., [32, 64, 128, 256] if start=32, depth=4

        # --- Bottleneck ---
        # Channels going into the bottleneck projection (output of last encoder block)
        self.bottleneck_in_chans = encoder_channels[-1] # e.g., 256
        # Spatial dim calculation should account for pooling happening cnn_depth - 1 times
        self.bottleneck_spatial_dim = img_size // (2**(cnn_depth - 1)) # spatial dim
        self.num_patches_bottleneck = self.bottleneck_spatial_dim ** 2   # Correct number of patches

        # Project encoder output to Transformer embedding dimension
        self.to_transformer_proj = nn.Conv2d(self.bottleneck_in_chans, transformer_embed_dim, kernel_size=1)

        # --- Transformer Bottleneck ---
        # Positional embedding for the flattened feature map
        self.transformer_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_bottleneck, transformer_embed_dim))
        # Layer index embedding
        self.transformer_layer_embed = nn.Embedding(num_layers_max, transformer_embed_dim)

        # Standard Transformer Encoder
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim, nhead=transformer_heads,
            dim_feedforward=transformer_embed_dim * 4, # Standard feedforward size
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True # Pre-LayerNorm is common now
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=transformer_layers,
            norm=nn.LayerNorm(transformer_embed_dim) # Final LayerNorm after stack
        )

        # --- Projection from Transformer ---
        # Project back from Transformer dimension to the channel count needed for the first decoder stage
        # The first decoder stage expects the same number of channels as the *last* encoder stage output
        self.from_transformer_proj = nn.Conv2d(transformer_embed_dim, self.bottleneck_in_chans, kernel_size=1)

        # --- CNN Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Input channels to the first decoder block is the output of the bottleneck projection
        current_chans = self.bottleneck_in_chans # e.g., 256

        # Iterate decoder stages (cnn_depth - 1 stages needed for upsampling)
        for i in range(cnn_depth - 1):
            skip_ch = self.encoder_channels[cnn_depth - 2 - i] # e.g., 128, 64, 32 for depth=4

            # Output channels for this decoder stage should match the skip connection channels
            out_ch = skip_ch
            # Add the UpsampleBlock
            self.decoder_blocks.append(
                UpsampleBlock(current_chans, skip_ch, out_ch) # In, Skip, Out
            )
            # Output channels of this block become input for the next
            current_chans = out_ch

        # Final Convolution: Map from last decoder block channels to output image channels
        # The last decoder block outputs channels matching the *first* encoder block (e.g., 32)
        self.final_conv = nn.Conv2d(current_chans, out_chans, kernel_size=1)
        # Final activation (Sigmoid for output in [0, 1])
        self.final_act = nn.Sigmoid()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Initialize model weights """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Kaiming init for Conv layers often works well with ReLU/GELU
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
             nn.init.normal_(m.weight, mean=0.0, std=0.02)


    def forward(self, x, layer_idx):
        """
        Forward pass through the hybrid model.
        Args:
            x (torch.Tensor): Input image batch [B, C_in, H, W]
            layer_idx (torch.Tensor): Layer indices for conditioning [B]
        Returns:
            torch.Tensor: Output image batch [B, C_out, H, W]
        """
        B = x.shape[0]
        skip_connections = []

        # --- CNN Encoding ---
        # Pass through encoder blocks, storing outputs for skip connections
        for i in range(self.cnn_depth):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            # Apply pooling after each block *except* the last one before bottleneck
            if i < self.cnn_depth - 1:
                 x = self.pool(x)
        # Note: The final 'x' here is the input to the bottleneck (e.g., [B, 256, H/16, W/16])

        # --- Transformer Bottleneck ---
        # Project to transformer dimension
        x_proj = self.to_transformer_proj(x) # [B, embed_dim, H_b, W_b]
        # Flatten spatial dimensions and transpose for Transformer (B, N, E)
        x_flat = x_proj.flatten(2).transpose(1, 2) # [B, H_b*W_b, embed_dim]
        # Get layer embeddings [B, embed_dim] -> [B, 1, embed_dim]
        l_embed = self.transformer_layer_embed(layer_idx).unsqueeze(1)
        l_embed_expanded = l_embed.expand(-1, self.num_patches_bottleneck, -1) # Expands dim 1

        x_tf_in = x_flat + self.transformer_pos_embed + l_embed_expanded
        x_tf_out = self.transformer_encoder(x_tf_in) # [B, H_b*W_b, embed_dim]

        # Reshape back to spatial format (B, E, H_b, W_b)
        x_tf_spatial = x_tf_out.transpose(1, 2).reshape(B, self.transformer_embed_dim, self.bottleneck_spatial_dim, self.bottleneck_spatial_dim)

        # Project back to CNN channel dimension for decoder
        x = self.from_transformer_proj(x_tf_spatial) # [B, bottleneck_in_chans, H_b, W_b]

        # --- CNN Decoding ---
        # Reverse skip connections for U-Net structure
        skip_connections = skip_connections[::-1]
        # Pass through decoder blocks, providing skip connections
        for i in range(self.cnn_depth - 1):
            current_skip_conn = skip_connections[i+1]
            # Pass current feature map 'x' and the corresponding skip connection
            x = self.decoder_blocks[i](x, current_skip_conn)

        # --- Final Output Layer ---
        x = self.final_conv(x) # Map to output channels
        output = self.final_act(x) # Apply sigmoid activation

        return output


# --- Training Function ---
def train_model(greyscale=True, subset_fraction=1.0):
    """ Trains the LayerGeneratorHybrid model """
    in_chans = 1 if greyscale else 4 # Input channels (1 for grayscale, 4 for RGBA assumed)
    out_chans = in_chans # Output channels usually match input

    print(f"--- Training Configuration ---")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Subset: {subset_fraction*100:.1f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"CNN Filters: {cnn_start_filters}, CNN Depth: {cnn_depth}")
    print(f"Transformer Dim: {transformer_embed_dim}, Layers: {transformer_layers}, Heads: {transformer_heads}")
    print(f"Loss Weights: MSE={lambda_mse}, Perceptual={lambda_perceptual}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    print(f"Loading dataset from: {data_file}")
    try:
        dataset = LayerPtDataset(data_file, img_size=img_size, transform=None, # Add transforms if needed
                                 greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")
        # Ensure dataset provides num_layers if needed by model conditioning
        if not hasattr(dataset, 'num_layers'):
             print("Warning: Dataset does not have 'num_layers' attribute. Using default max_layers=18 for embedding.")
             num_layers_max = 18
        else:
             num_layers_max = dataset.num_layers
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True) # num_workers > 0 for background loading
    print(f"Dataloader created. Batches per epoch: {len(dataloader)}")

    # **** Instantiate the Hybrid Model ****
    model = LayerGeneratorHybrid(
        img_size=img_size, in_chans=in_chans, out_chans=out_chans,
        cnn_start_filters=cnn_start_filters, cnn_depth=cnn_depth, # Use defined hyperparameters
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_layers_max=num_layers_max # Get from dataset or default
    )

    # Setup device, model, optimizer, losses, scheduler, scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")

    # --- Loss Functions ---
    criterion_mse = WeightedProportionalMSELoss(threshold=0.05).to(device)
    # VGG layers indices based on vgg19.features structure (approx after ReLU)
    # Example: 2=relu1_1, 7=relu2_1, 16=relu3_1, 25=relu4_1, 34=relu5_1
    criterion_perceptual = VGGPerceptualLoss(feature_layers=[2, 7, 16, 25, 34]).to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = num_epochs * len(dataloader)
    # Cosine annealing scheduler: warms down LR towards eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6) # Ensure T_max > 0
    # Automatic Mixed Precision (AMP) scaler for potential speedup on CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss_total = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_perc = 0.0
        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Ensure data unpacking matches dataset __getitem__
                input_image, target, layer_idx = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch data at index {batch_idx}. Expected 3 items (input, target, layer_idx). Got: {len(batch_data)}. Error: {e}")
                continue # Skip this batch

            # Move data to the training device
            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)

            # Reset gradients before forward pass
            optimizer.zero_grad(set_to_none=True) # More memory efficient

            # --- Forward pass with AMP context ---
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                output = model(input_image, layer_idx)

                # --- Shape Check and Potential Resize ---
                # Ensure output and target shapes match before loss calculation
                if output.shape != target.shape:
                    # Check if only spatial dimensions differ (H, W)
                    if output.shape[0:2] == target.shape[0:2] and len(output.shape) == 4 and len(target.shape) == 4:
                        print(f"\nWarning: Resizing model output {output.shape} to match target {target.shape} at batch {batch_idx}.")
                        output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
                    else:
                        # If batch size or channels differ, it's a critical error
                        print(f"\nError: Unrecoverable shape mismatch! Output: {output.shape}, Target: {target.shape}. Skipping batch {batch_idx}.")
                        continue # Skip this batch

                # --- Calculate Loss Components ---
                loss_mse = criterion_mse(input_image, output, target)
                loss_perceptual = criterion_perceptual(output, target) # Compare generated vs target

                # --- Combine Losses with Weights ---
                loss = (lambda_mse * loss_mse) + (lambda_perceptual * loss_perceptual)

            # --- Backward pass and Optimization ---
            # Check for NaN/Inf loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                # Optionally log which component was NaN/Inf
                if torch.isnan(loss_mse) or torch.isinf(loss_mse): print("  MSE Loss was NaN/Inf")
                if torch.isnan(loss_perceptual) or torch.isinf(loss_perceptual): print("  Perceptual Loss was NaN/Inf")
                # Reset gradients and skip optimizer step for this batch
                optimizer.zero_grad(set_to_none=True)
                continue

            # Scale the loss for AMP
            scaler.scale(loss).backward()
            # Unscale gradients and step optimizer
            scaler.step(optimizer)
            # Update the scaler for next iteration
            scaler.update()
            # Step the learning rate scheduler
            scheduler.step()

            # --- Logging ---
            # Accumulate losses for epoch average
            epoch_loss_total += loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_perc += loss_perceptual.item()
            # Update progress bar postfix with current batch losses and LR
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                perc=f"{loss_perceptual.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.1e}" # Get current LR from scheduler
            )
            # --- End of Batch Loop ---

        # --- End of Epoch ---
        progress_bar.close() # Close the tqdm bar for the epoch
        if len(dataloader) > 0:
            # Calculate average losses for the epoch
            avg_epoch_loss = epoch_loss_total / len(dataloader)
            avg_mse_loss = epoch_loss_mse / len(dataloader)
            avg_perc_loss = (epoch_loss_perc / len(dataloader)) * lambda_perceptual
            current_lr = scheduler.get_last_lr()[0]
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} Summary -> Avg Loss: {avg_epoch_loss:.4f} "
                  f"(MSE: {avg_mse_loss:.4f}, Perc: {avg_perc_loss:.4f}) - LR: {current_lr:.2e}")
        else:
            # Handle case where dataloader might be empty
            print(f"Epoch {epoch+1}/{num_epochs} - No batches processed.")

        # --- Visualization and Checkpointing ---
        # Perform at specified frequency or on the last epoch
        if (epoch + 1) % VISUALIZE_AND_CHECKPOINT_FREQUENCY == 0 or epoch == num_epochs - 1:
            print(f"--- Running Visualization & Checkpointing for Epoch {epoch+1} ---")
            try:
                # Use last batch data for visualization (or load a fixed validation batch)
                # Ensure the batch data exists (in case the last batch was skipped)
                if 'input_image' in locals() and 'target' in locals() and 'output' in locals():
                    # No need to move to device again, already there
                    # Set model to evaluation mode for consistent visualization
                    model.eval()
                    with torch.no_grad(): # No gradients needed for visualization inference
                         # Rerun inference if needed, or use the 'output' from the last batch
                         # Using 'output' directly might be slightly off if BN stats changed
                         # Safer to re-run with eval mode:
                         with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                              output_vis = model(input_image, layer_idx) # Use last batch input/layer_idx

                         # Save image grid
                         save_image_grid(input_image, target, output_vis, epoch, save_dir=save_dir)
                         print(f"Visualization grid saved for epoch {epoch+1}")

                    # Set model back to training mode
                    model.train()

                else:
                     print("Skipping visualization: Last batch data not available.")


                # --- Save Model Checkpoint ---
                if SAVE_MODEL:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
                    # Save model state, optimizer, scheduler, epoch, and losses
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_epoch_loss, # Save average epoch loss
                        'loss_mse': avg_mse_loss,
                        'loss_perceptual': avg_perc_loss,
                        'hyperparameters': { # Optionally save hyperparameters for reproducibility
                             'img_size': img_size, 'batch_size': batch_size, 'lr': learning_rate,
                             'lambda_mse': lambda_mse, 'lambda_perceptual': lambda_perceptual,
                             # Add other relevant hyperparameters
                        }
                    }, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                print(f"\nError during visualization/checkpointing at epoch {epoch+1}: {e}")
                # Ensure model is back in train mode even if an error occurs
                model.train()
            print("-" * 20) # Separator after checkpointing info
            # --- End Checkpointing Block ---

    # --- End of Training Loop ---
    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print("-" * 50)
    print(f"Training completed in {total_training_time / 60:.2f} minutes ({total_training_time:.1f} seconds).")

    # Save the final model state dictionary
    if SAVE_MODEL:
        final_model_path = os.path.join(save_dir, 'hybrid_layer_generator_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state_dict saved as {final_model_path}")

    print(f"Results (checkpoints, visualizations) saved in: {save_dir}")
    print("-" * 50)

# --- Main Execution Guard ---
if __name__ == '__main__':
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Start Training ---
    # Consider wrapping in try...except for robust execution
    try:
        train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    except Exception as main_e:
        print(f"\n--- An error occurred during the training process ---")
        print(f"Error Type: {type(main_e).__name__}")
        print(f"Error Details: {main_e}")
        # Optionally add more detailed traceback logging here if needed
        # import traceback
        # print("\n--- Traceback ---")
        # traceback.print_exc()
        print("-" * 50)

