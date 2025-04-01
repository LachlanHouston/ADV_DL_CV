import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """Encoder network to extract features from input image."""
    def __init__(self, in_channels=1):
        super().__init__()
        
        # Encoder blocks (downsampling)
        self.enc1 = ConvBlock(in_channels, 32)                  # 64x64 -> 64x64
        self.enc2 = ConvBlock(32, 64)                           # 64x64 -> 64x64
        self.pool1 = nn.MaxPool2d(2)                            # 64x64 -> 32x32
        
        self.enc3 = ConvBlock(64, 64)                           # 32x32 -> 32x32
        self.enc4 = ConvBlock(64, 128)                          # 32x32 -> 32x32
        self.pool2 = nn.MaxPool2d(2)                            # 32x32 -> 16x16
        
        self.enc5 = ConvBlock(128, 128)                         # 16x16 -> 16x16
        self.enc6 = ConvBlock(128, 256)                         # 16x16 -> 16x16
        self.pool3 = nn.MaxPool2d(2)                            # 16x16 -> 8x8
        
        self.enc7 = ConvBlock(256, 256)                         # 8x8 -> 8x8
        self.enc8 = ConvBlock(256, 512)                         # 8x8 -> 8x8
        
    def forward(self, x):
        # Encoding path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x = self.pool1(x2)
        
        x3 = self.enc3(x)
        x4 = self.enc4(x3)
        x = self.pool2(x4)
        
        x5 = self.enc5(x)
        x6 = self.enc6(x5)
        x = self.pool3(x6)
        
        x7 = self.enc7(x)
        x8 = self.enc8(x7)
        
        # Return encoded features and skip connections
        return x8, (x6, x4, x2)


class LayerEmbedding(nn.Module):
    """Layer position embedding to condition the generation."""
    def __init__(self, num_layers=18, embedding_dim=512, output_size=8):
        super().__init__()
        self.embedding = nn.Embedding(num_layers, embedding_dim)
        
        # Convert embedding to spatial features
        self.to_spatial = nn.Sequential(
            nn.Linear(embedding_dim, output_size * output_size),
            nn.ReLU(inplace=True)
        )
        
        self.output_size = output_size
        
    def forward(self, layer_idx):
        # Get embedding vector
        embed = self.embedding(layer_idx)  # [batch_size, embedding_dim]
        
        # Convert to spatial features
        spatial = self.to_spatial(embed)  # [batch_size, output_size*output_size]
        
        # Reshape to spatial feature map
        batch_size = embed.shape[0]
        return spatial.view(batch_size, 1, self.output_size, self.output_size)


class Decoder(nn.Module):
    """Decoder network to generate output image."""
    def __init__(self, out_channels=1):
        super().__init__()
        
        # Decoder blocks (upsampling)
        self.dec1 = ConvBlock(512 + 1, 256)  # +1 for layer embedding
        self.dec2 = ConvBlock(256, 256)
        
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(256 + 256, 128)  # +256 for skip connection
        self.dec4 = ConvBlock(128, 128)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec5 = ConvBlock(128 + 128, 64)  # +128 for skip connection
        self.dec6 = ConvBlock(64, 64)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec7 = ConvBlock(64 + 64, 32)  # +64 for skip connection
        self.dec8 = ConvBlock(32, 32)
        
        # Final layer to produce grayscale output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x, layer_embed, skip_connections):
        # Bottleneck features with layer embedding
        x = torch.cat([x, layer_embed], dim=1)
        x = self.dec1(x)
        x = self.dec2(x)
        
        # Decoding path with skip connections
        x = self.upsample1(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.dec3(x)
        x = self.dec4(x)
        
        x = self.upsample2(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.dec5(x)
        x = self.dec6(x)
        
        x = self.upsample3(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.dec7(x)
        x = self.dec8(x)
        
        # Final convolution and sigmoid to bound outputs between 0 and 1
        return torch.sigmoid(self.final(x))


class AutoregressiveCharacterGenerator(nn.Module):
    """
    Autoregressive model for grayscale character generation.
    Works in pixel space with 64x64 images, generating one layer at a time.
    """
    def __init__(self, num_layers=18):
        super().__init__()
        
        self.encoder = Encoder(in_channels=1)  # Grayscale input
        self.layer_embedding = LayerEmbedding(num_layers=num_layers, output_size=8)
        self.decoder = Decoder(out_channels=1)  # Grayscale output
        self.num_layers = num_layers
        
    def forward(self, x, layer_idx):
        """
        Forward pass through the model.
        
        Args:
            x: Input image [batch_size, 1, 64, 64]
            layer_idx: Layer index to generate (0-17)
            
        Returns:
            Generated layer as [batch_size, 1, 64, 64]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Convert layer_idx to tensor if it's not already
        if isinstance(layer_idx, int):
            layer_idx = torch.tensor([layer_idx], device=device).expand(batch_size)
        
        # Encode input image
        features, skip_connections = self.encoder(x)
        
        # Get layer embedding
        layer_embed = self.layer_embedding(layer_idx)
        
        # Decode to output image
        output = self.decoder(features, layer_embed, skip_connections)
        
        return output
    
    def generate_sequence(self, seed_image=None, num_layers=None):
        """
        Generate a full sequence of layers starting from a seed image.
        
        Args:
            seed_image: Initial image (typically blank). If None, creates a blank image.
            num_layers: Number of layers to generate. If None, uses self.num_layers.
            
        Returns:
            Tensor: Generated sequence of layers [num_layers, 1, 64, 64]
        """
        device = next(self.parameters()).device
        num_layers = num_layers or self.num_layers
        
        # Create a blank canvas if no seed is provided
        if seed_image is None:
            current_image = torch.zeros(1, 1, 64, 64, device=device)
        else:
            current_image = seed_image.unsqueeze(0).to(device)
            
        generated_layers = [current_image.squeeze(0)]
        
        # Generate layers one by one
        for i in range(1, num_layers):  # Start from 1 if we consider the seed as layer 0
            with torch.no_grad():
                # Generate next layer
                next_layer = self.forward(current_image, i-1)  # i-1 because we're predicting the next layer
                generated_layers.append(next_layer.squeeze(0))
                current_image = next_layer
                
        # Stack all generated layers
        return torch.stack(generated_layers)