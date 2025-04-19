"""
Contains helper functions for visualizing model outputs, including
saving image grids and generating layer-by-layer rollouts.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    n_images = min(15, input_images.shape[0]) # Limit grid width
    fig, axes = plt.subplots(3, n_images, figsize=(2 * n_images, 6)) # Rows: Input, Target, Predicted

    # Ensure axes is always 2D array for consistent indexing
    if n_images == 1:
        axes = axes.reshape(3, 1)

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
        if i == 0: ax.set_ylabel('Input', rotation=0, size='large', labelpad=30)


        # Target Image
        img_target, cmap_target = convert_for_imshow(target_images[i])
        ax = axes[1, i]
        ax.imshow(img_target, cmap=cmap_target)
        ax.axis('off')
        if i == 0: ax.set_ylabel('Target', rotation=0, size='large', labelpad=30)

        # Predicted Image
        img_pred, cmap_pred = convert_for_imshow(predicted_images[i])
        ax = axes[2, i]
        ax.imshow(img_pred, cmap=cmap_pred)
        ax.axis('off')
        if i == 0: ax.set_ylabel('Predicted', rotation=0, size='large', labelpad=30)

        # Add title to the top row only
        if i == n_images // 2: # Center the epoch title roughly
             axes[0, i].set_title(f'Epoch {epoch+1}', pad=20)


    plt.tight_layout(pad=0.1, h_pad=0.5) # Adjust padding
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch+1:03d}.png'), dpi=150) # Adjust DPI if needed
    plt.close(fig) # Close the figure to free memory


def _rollout(model, img_size, in_chans, num_layers=18,
             device='cpu', start_canvas=None):
    """
    Generate a full sequence layer‑by‑layer.
    If `start_canvas` is supplied it’s used as layer‑0; otherwise start from zeros.
    Returns a list of (C,H,W) tensors including the starting canvas.
    """
    if start_canvas is None:
        canvas = torch.zeros(1, in_chans, img_size, img_size, device=device)
    else:
        # Ensure start_canvas has batch dim [1, C, H, W] and is on correct device
        if start_canvas.dim() == 3:
            start_canvas = start_canvas.unsqueeze(0)
        canvas = start_canvas.clone().to(device) # [1,C,H,W]

    seq = [canvas.squeeze(0).cpu()]          # store layer‑0
    model.eval() # Ensure model is in eval mode for rollout
    with torch.no_grad():
        for l in range(1, num_layers):
            idx = torch.tensor([l], device=device)
            canvas = model(canvas, idx) # Generate next layer
            seq.append(canvas.squeeze(0).cpu()) # Store result
    return seq

def _sample_first_layer(dataset, device):
    """
    Grab a random sample whose layer_idx == 0 from the dataset and return its
    input-canvas tensor. Falls back to any sample if none are found.
    """
    idxs = list(range(len(dataset)))
    random.shuffle(idxs)
    for i in idxs:
        try:
            # Try unpacking assuming (input, target, layer_idx) structure
            inp, _, layer_idx = dataset[i]
            if int(layer_idx) == 0:
                return inp.to(device)
        except (ValueError, TypeError):
            # Handle cases where dataset[i] might not return 3 items or layer_idx isn't right type
             try: # Maybe it's just (input, layer_idx)? Adapt if needed based on dataset
                 inp, layer_idx = dataset[i]
                 if int(layer_idx) == 0:
                     return inp.to(device)
             except:
                 continue # Skip if unpacking fails

    # Fallback: return the input of the very first sample
    print("Warning: Could not find a sample with layer_idx == 0. Using the first sample's input for rollout start.")
    try:
        inp, _, _ = dataset[0] # Assume 3 items again for fallback
        return inp.to(device)
    except: # Final fallback if even the first sample fails unpacking
        print("Error: Failed to get any sample from the dataset for rollout.")
        return torch.zeros(1, dataset.in_chans, dataset.img_size, dataset.img_size, device=device) # Adjust shape details if possible


def save_rollout_grid(model, save_dir, img_size, in_chans,
                      device, epoch, dataset,
                      n_rollouts=5, num_layers=18):
    """Rows = roll‑outs, Cols = layers. PNG written next to checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    rows, cols = n_rollouts, num_layers
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))

    # Ensure axes is always 2D array for consistent indexing
    if rows == 1 and cols == 1:
         axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, cols)
    elif cols == 1:
        axes = axes.reshape(rows, 1)


    for r in range(rows):
        start_canvas = _sample_first_layer(dataset, device)
        seq = _rollout(model, img_size, in_chans, num_layers, device, start_canvas)
        for c, img in enumerate(seq):
            if c >= cols: break # Don't try to plot more columns than exist
            ax = axes[r, c]
            # --- convert tensor to displayable array with correct channel order ---
            array_img, cmap = convert_for_imshow(img)
            if array_img is not None:
                if cmap == 'gray':
                    ax.imshow(array_img, cmap='gray', vmin=0, vmax=1) # Ensure range for grayscale
                else:
                    ax.imshow(array_img) # Assumes RGB/RGBA in [0,1]
            ax.axis('off')
            if r == 0:
                ax.set_title(f"L{c}", fontsize=8, pad=2) # Adjust font size/padding

    plt.subplots_adjust(wspace=0.05, hspace=0.05) # Reduce spacing between images
    out_path = os.path.join(save_dir, f'rollouts_epoch_{epoch+1:03d}.png') # Match epoch numbering with grid
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Roll-out grid saved to {out_path}")