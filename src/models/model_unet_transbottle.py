"""
Contains the LayerGeneratorHybrid model definition, its building blocks,
and the VGGPerceptualLoss class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.functional import normalize

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
        # NOTE: Corrected pooling calculation - pooling happens 'cnn_depth' times to reach bottleneck
        self.bottleneck_spatial_dim = img_size // (2**cnn_depth) # spatial dim after pooling 'cnn_depth' times
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
        self.from_transformer_proj = nn.Conv2d(transformer_embed_dim, self.bottleneck_in_chans, kernel_size=1)

        # --- CNN Decoder ---
        self.decoder_blocks = nn.ModuleList()
        # Input channels to the first decoder block is the output of the bottleneck projection
        current_chans = self.bottleneck_in_chans # e.g., 256

        # Iterate decoder stages (cnn_depth stages needed for upsampling)
        for i in range(cnn_depth):
            # Determine skip connection channels (reverse order from encoder, skipping bottleneck output)
            # The last decoder block (i=cnn_depth-1) connects to the first encoder block output (index 0)
            skip_idx = self.cnn_depth - 1 - i
            skip_ch = self.encoder_channels[skip_idx] # e.g., for i=0, skip_idx=3 -> 256; for i=3, skip_idx=0 -> 32

            # Output channels for this decoder stage should match the skip connection channels being connected TO
            # The final output before the 1x1 conv should match the first encoder block's channels
            out_ch = skip_ch

            # Add the UpsampleBlock
            self.decoder_blocks.append(
                UpsampleBlock(current_chans, skip_ch, out_ch) # In, Skip, Out
            )
            # Output channels of this block become input for the next
            current_chans = out_ch # This is now, e.g., 32 after the last block

        # Final Convolution: Map from last decoder block channels to output image channels
        self.final_conv = nn.Conv2d(current_chans, out_chans, kernel_size=1)
        # Final activation (Sigmoid for output in [0, 1])
<<<<<<< Updated upstream
        # self.final_act = nn.Sigmoid()
        # self.final_act = nn.Tanh()
        self.final_act = nn.Identity()
=======
        # self.final_act = nn.Tanh()
        self.final_act = nn.Sigmoid()
        
>>>>>>> Stashed changes

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
        current_feat = x
        for i in range(self.cnn_depth):
            current_feat = self.encoder_blocks[i](current_feat)
            skip_connections.append(current_feat)
            # Apply pooling *after* each block
            current_feat = self.pool(current_feat)
        # Note: The final 'current_feat' here is the input to the bottleneck (e.g., [B, 256, H/16, W/16])
        # It has been pooled cnn_depth times.

        # --- Transformer Bottleneck ---
        # Project to transformer dimension
        x_proj = self.to_transformer_proj(current_feat) # [B, embed_dim, H_b, W_b]
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
        x_dec = self.from_transformer_proj(x_tf_spatial) # [B, bottleneck_in_chans, H_b, W_b]

        # --- CNN Decoding ---
        # Reverse skip connections for U-Net structure (we need skip_connections[cnn_depth-1] down to skip_connections[0])
        skip_connections = skip_connections[::-1] # Now indexed 0..cnn_depth-1 (e.g., 0 is last encoder output, 3 is first)
        # Pass through decoder blocks, providing skip connections
        for i in range(self.cnn_depth):
            # The i-th decoder block uses the i-th skip connection from the reversed list
            current_skip_conn = skip_connections[i]
            # Pass current feature map 'x_dec' and the corresponding skip connection
            x_dec = self.decoder_blocks[i](x_dec, current_skip_conn)

        # --- Final Output Layer ---
        x_final = self.final_conv(x_dec) # Map to output channels
        output = self.final_act(x_final) # Apply sigmoid activation

        return output


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