import torch
import matplotlib.pyplot as plt
import os, sys

# --- Ensure project structure allows imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data import LayerPtDataset


vis_3_3 = False

def bgr_to_rgb(t):
    """
    t: Tensor C×H×W, C=3 (BGR) or C=4 (BGRA)
    Returns tensor with channels reordered to RGB / RGBA.
    """
    if t.shape[0] == 3:          # B,G,R  ->  R,G,B
        return t[[2, 1, 0], ...]
    elif t.shape[0] == 4:        # B,G,R,A -> R,G,B,A
        return t[[2, 1, 0, 3], ...]
    return t                     # single-channel etc.

# 1) Load the dataset
pt_path = 'data/full_dataset_128.pt'
ds = LayerPtDataset(pt_path, img_size=64, greyscale=False)

# 2) Grab layers for the first sample; make sure they’re C×H×W floats in [0,1]
sample = ds.all_data[0]
if isinstance(sample, torch.Tensor) and sample.dtype == torch.uint8 and sample.shape[-1] in (1,3,4):
    sample = sample.permute(0, 3, 1, 2).float() / 255.0
elif isinstance(sample, torch.Tensor) and sample.dtype != torch.uint8 and sample.shape[1] in (1,3,4):
    pass  # already CHW float
else:
    sample = torch.from_numpy(sample).float() / 255.0
    if sample.shape[-1] in (1,3,4):
        sample = sample.permute(0, 3, 1, 2)

n_layers = sample.shape[0]

if vis_3_3:
    first_idxs = [0, 1, 2]
    last_idxs  = [n_layers-3, n_layers-2, n_layers-1]

    # 3) Plot first 3 … last 3
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.5))
    for ax in axes: ax.axis("off")

    # first three
    for col, li in enumerate(first_idxs):
        img = bgr_to_rgb(sample[li]).permute(1, 2, 0).numpy()  # H×W×C, RGB now
        axes[col].imshow(img)
        axes[col].set_title(f"Layer {li+1}", fontsize=8)

    # ellipsis
    axes[3].text(0.5, 0.5, "…", ha="center", va="center", fontsize=24)

    # last three
    for i, li in enumerate(last_idxs, start=4):
        img = bgr_to_rgb(sample[li]).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"Layer {li+1}", fontsize=8)

    plt.tight_layout()
    plt.savefig("layer_3_3_visualization.png", dpi=300)

else:
    fig, axes = plt.subplots(
        2, 9, 
        figsize=(12, 4),
        gridspec_kw={'wspace': -0.1, 'hspace': -0.4}  # tighten horizontal spacing
    )

    for row in range(2):
        for col in range(9):
            idx = row * 9 + col
            if idx < n_layers:
                img = bgr_to_rgb(sample[idx]).permute(1, 2, 0).numpy()
                axes[row, col].imshow(img)
                axes[row, col].set_title(f"Layer {idx+1}", fontsize=8)
            axes[row, col].axis("off")

    plt.savefig("layer_all_visualization.png", dpi=300)