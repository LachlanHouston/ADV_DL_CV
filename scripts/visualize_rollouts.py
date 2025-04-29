# visualize_rollouts.py
import os, math
import torch, matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict
import re

# ----------------- User configuration -----------------
CKPT_PATH = 'results/hybrid_percLoss_20250419_175849/checkpoint_epoch_010.pt'  # set your checkpoint path
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
ROLLOUTS   = 5    # number of independent roll‑outs
NUM_LAYERS = 18   # layers per roll‑out
# ------------------------------------------------------

# ---------- Model definition (exactly the same as in the training file) ----------
#   –– trimmed a bit: weight‑init helpers removed for brevity ––
import torch.nn as nn
class ConvBlock(nn.Module):
    """
    Matches the naming scheme of the training script:
    attributes are called `conv`, `bn`, and `act` instead of an unnamed Sequential.
    """
    def __init__(self, ic, oc, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(oc)
        self.act  = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat([x,skip],1)
        return self.conv(x)

class LayerGeneratorHybrid(nn.Module):
    def __init__(self, img_size=64, in_chans=4, out_chans=4,
                 cnn_start_filters=32, cnn_depth=4,
                 transformer_embed_dim=256, transformer_layers=6,
                 transformer_heads=8, num_layers_max=18):
        super().__init__()
        self.cnn_depth = cnn_depth
        # ----- encoder -----
        self.encoder, ch, enc_ch = nn.ModuleList(), in_chans, []
        for d in range(cnn_depth):
            self.encoder.append(nn.Sequential(ConvBlock(ch, cnn_start_filters<<d),
                                              ConvBlock(cnn_start_filters<<d,
                                                        cnn_start_filters<<d)))
            enc_ch.append(cnn_start_filters<<d)
            ch = cnn_start_filters<<d
        self.pool = nn.MaxPool2d(2)
        # ----- bottleneck -----
        h = img_size // (2 ** (cnn_depth-1))
        self.to_tf   = nn.Conv2d(ch, transformer_embed_dim, 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, h*h, transformer_embed_dim))
        self.layer_emb = nn.Embedding(num_layers_max, transformer_embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_embed_dim * 4,  # 256 → 1024 just like during training
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation='gelu'
        )
        self.tf = nn.TransformerEncoder(enc_layer, transformer_layers,
                                        norm=nn.LayerNorm(transformer_embed_dim))
        self.from_tf = nn.Conv2d(transformer_embed_dim, ch, 1)
        # ----- decoder -----
        self.decoder = nn.ModuleList()
        for d in range(cnn_depth-1):
            skip = enc_ch[-2-d]
            self.decoder.append(UpsampleBlock(ch, skip, skip))
            ch = skip
        self.final = nn.Sequential(nn.Conv2d(ch, out_chans, 1), nn.Sigmoid())

    def forward(self, x, layer_idx):
        skips=[]
        for d,blk in enumerate(self.encoder):
            x = blk(x); skips.append(x)
            if d<self.cnn_depth-1: x = self.pool(x)
        h = x.shape[-1]
        x = self.to_tf(x).flatten(2).transpose(1,2)
        x = x + self.pos_emb + self.layer_emb(layer_idx).unsqueeze(1).expand_as(x)
        x = self.tf(x).transpose(1,2).reshape(x.shape[0], -1, h, h)
        x = self.from_tf(x)
        for blk,skip in zip(self.decoder, skips[::-1][1:]): x = blk(x,skip)
        return self.final(x)
# -------------------------------------------------------------------------------


def rollout(model, img_size, in_chans, num_layers=18, device='cpu'):
    canvas = torch.zeros(1, in_chans, img_size, img_size, device=device)
    layers = []
    model.eval()
    with torch.no_grad():
        for l in range(num_layers):
            idx = torch.tensor([l], device=device)
            canvas = model(canvas, idx)
            layers.append(canvas.squeeze(0).cpu())
    return layers  # list of (C,H,W) tensors


def visualize(seqs, save_path=None):
    rows, cols = len(seqs), len(seqs[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))
    for r, seq in enumerate(seqs):
        for c, img in enumerate(seq):
            ax = axes[r, c] if rows>1 else axes[c]
            np_img = img.permute(1,2,0).clip(0,1).numpy()
            cmap = 'gray' if np_img.shape[2]==1 else None
            ax.imshow(np_img.squeeze() if cmap else np_img, cmap=cmap)
            ax.set_axis_off()
            if r==0: ax.set_title(f"L{c}", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    # plt.show()


def main():
    
    # Removed argparse block - using configuration constants instead

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    hp = ckpt.get('hyperparameters', {})  # training script stores these
    img_size  = hp.get('img_size', 64)
    in_chans  = hp.get('in_chans', 4)
    model = LayerGeneratorHybrid(
        img_size=img_size,
        in_chans=in_chans,
        out_chans=in_chans,
        cnn_start_filters=hp.get('cnn_start_filters', 32),
        cnn_depth=hp.get('cnn_depth', 4),
        transformer_embed_dim=hp.get('transformer_embed_dim', 256),
        transformer_layers=hp.get('transformer_layers', 6),
        transformer_heads=hp.get('transformer_heads', 8),
        num_layers_max=hp.get('num_layers_max', 18)
    )
    # --- alias names to match the original training script ---
    # (Removed unused alias assignments)

    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    # --- remap old parameter names to new model names ---
    rename_rules = [
        (r'^encoder_blocks\.', 'encoder.'),
        (r'^decoder_blocks\.', 'decoder.'),
        (r'^transformer_pos_embed', 'pos_emb'),
        (r'^transformer_layer_embed', 'layer_emb'),
        (r'^to_transformer_proj\.', 'to_tf.'),
        (r'^transformer_encoder\.', 'tf.'),
        (r'^from_transformer_proj\.', 'from_tf.'),
        (r'^final_conv\.', 'final.0.')
    ]
    new_state = OrderedDict()
    for k, v in state.items():
        new_k = k
        for pattern, repl in rename_rules:
            new_k = re.sub(pattern, repl, new_k)
        new_state[new_k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"Warning: {len(missing)} keys not found in checkpoint.")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected keys in checkpoint.")

    model.to(DEVICE)

    seqs = [rollout(model, img_size, in_chans, NUM_LAYERS, DEVICE)
            for _ in range(ROLLOUTS)]
    out_img = Path(CKPT_PATH).with_suffix('.rollouts.png')
    visualize(seqs, out_img)
    print(f"Saved grid to {out_img.resolve()}")

if __name__ == '__main__':
    main()