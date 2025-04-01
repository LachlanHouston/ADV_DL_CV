import torch
from torch import nn
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
import tqdm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

def positional_encoding_2d(nph, npw, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(nph), torch.arange(npw), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0, f'Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})'
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.k_projection  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_projeciton  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):

        batch_size, seq_len, embed_dim = x.size()
        keys    = self.k_projection(x)
        queries = self.q_projection(x)
        values  = self.v_projeciton(x)

        # Rearrange keys, queries and values 
        # from batch_size x seq_len x embed_dim to (batch_size x num_head) x seq_len x head_dim
        keys = rearrange(keys, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        queries = rearrange(queries, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)
        values = rearrange(values, 'b s (h d) -> (b h) s d', h=self.num_heads, d=self.head_dim)

        attention_logits = torch.matmul(queries, keys.transpose(1, 2))
        attention_logits = attention_logits * self.scale
        attention = F.softmax(attention_logits, dim=-1)
        out = torch.matmul(attention, values)

        # Rearragne output
        # from (batch_size x num_head) x seq_len x head_dim to batch_size x seq_len x embed_dim
        out = rearrange(out, '(b h) s d -> b s (h d)', h=self.num_heads, d=self.head_dim)

        assert attention.size() == (batch_size*self.num_heads, seq_len, seq_len)
        assert out.size() == (batch_size, seq_len, embed_dim)

        return self.o_projection(out)

class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_dim=None, dropout=0.0):
        super().__init__()

        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

        fc_hidden_dim = 4*embed_dim if fc_dim is None else fc_dim

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, fc_hidden_dim),
            nn.GELU(),
            nn.Linear(fc_hidden_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.layernorm1(attention_out + x)
        x = self.dropout(x)
        fc_out = self.fc(x)
        x = self.layernorm2(fc_out + x)
        x = self.dropout(x)
        return x

class ViT(nn.Module):
    def __init__(self, image_size, channels, patch_size, embed_dim, num_heads, num_layers,
                 pos_enc='fixed', pool='cls', dropout=0.0, 
                 fc_dim=None):
        
        super().__init__()

        assert pool in ['cls', 'mean', 'max']
        assert pos_enc in ['fixed', 'learnable']

        self.pool, self.pos_enc, = pool, pos_enc

        H, W = image_size
        patch_h, patch_w = patch_size
        assert H % patch_h == 0 and W % patch_w == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (H // patch_h) * (W // patch_w)
        patch_dim = channels * patch_h * patch_w

        if self.pool == 'cls':
            self.cls_token = nn.Parameter(torch.rand(1,1,embed_dim))
            num_patches += 1
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        if self.pos_enc == 'learnable':
            self.positional_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        elif self.pos_enc == 'fixed':
            self.positional_embedding = positional_encoding_2d(
                nph = H // patch_h, 
                npw = W // patch_w,
                dim = embed_dim,
            )  

        transformer_blocks = []
        for i in range(num_layers):
            transformer_blocks.append(
                EncoderBlock(embed_dim=embed_dim, num_heads=num_heads, fc_dim=fc_dim, dropout=dropout))

        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.dropout = nn.Dropout(dropout)

        # Project 768 to 256 * 14 * 14 = 50176
        self.fc = nn.Linear(embed_dim, 256 * 14 * 14)

        self.decoder = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [256, 14, 14] -> [256, 28, 28]
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [128, 28, 28] -> [128, 56, 56]
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [64, 56, 56] -> [64, 112, 112]
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # [32, 112, 112] -> [32, 224, 224]
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
            nn.Tanh()  # or Sigmoid, depending on how you scale your images
        )


    def forward(self, img):

        tokens = self.to_patch_embedding(img)
        batch_size, num_patches, embed_dim = tokens.size()
        
        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '1 1 e -> b 1 e', b=batch_size)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            num_patches+=1
        
        positions =  self.positional_embedding.to(img.device, dtype=img.dtype)
        if self.pos_enc == 'fixed' and self.pool=='cls':
            positions = torch.cat([torch.zeros(1, embed_dim).to(img.device), positions], dim=0)
        x = tokens + positions
        
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        
        if self.pool =='max':
            x_pooled = x.max(dim=1)[0]
        elif self.pool =='mean':
            x_pooled = x.mean(dim=1)
        elif self.pool == 'cls':
            x_pooled = x[:, 0]

        x_proj = self.fc(x_pooled)
        x_img = self.decoder(x_proj.view(batch_size, 256, 14, 14))

        return x_img
    
if __name__ == '__main__':
    # Lav ny fil med train kode og data loader
    model = ViT(image_size=(28, 28), channels=1, patch_size=(7, 7), embed_dim=768, num_heads=4, num_layers=4, pos_enc='learnable', pool='cls', dropout=0.1)
    #img = torch.randn(32, 6, 4, 224, 224)

    # Loss function
    loss_function = nn.MSELoss()

    # Optimizer
    opt = torch.optim.AdamW(lr=1e-4, params=model.parameters(), weight_decay=1e-4)
    
    kenny_dataset = h5py.File('data/kenney_dataset_1000.h5', 'r') # [num_images, num_layers, pixel_h, pixel_w, num_channels]
    print(kenny_dataset['images'].shape)

    # Create a DataLoader from kenny_dataset
    class KenneyDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]
            # Transform to 64x64 greyscale
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(0, 3, 1, 2)
            images = torch.zeros((image.shape[0], 1, 28, 28), dtype=torch.float32)
            for i in range(image.shape[0]):
                resized_img = F.interpolate(image[i].unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False)
                # Flatten to greyscale
                greyscaled_img = resized_img.mean(dim=1, keepdim=True)
                # Normalize to [0, 1]
                images[i] = (greyscaled_img - greyscaled_img.min()) / (greyscaled_img.max() - greyscaled_img.min())

            return images
        
    # Create the dataset and dataloader
    dataset = KenneyDataset(kenny_dataset['images'])
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Print the shape of the images in the dataset
    print(f"Dataset shape: {dataset[0].shape}")

    # # Print the shape of the images in the dataloader
    # print(f"Dataloader shape: {next(iter(train_loader))[0].shape}")

    # Train the model
    num_epochs = 1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for i, images in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            images = images.squeeze(0)
            for j in range(images.shape[0]):
                images[j] = images[j]

        print(f"Loss: {running_loss/len(train_loader)}")