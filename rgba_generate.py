import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from PIL import Image
import io

from src.data.rgba_character_dataset import FastRgbaPtDataset
import pdb

# Import the RGBA model
from src.models.rgba_ar_cnn_model import AutoregressiveRGBAGenerator

def rgba_to_rgb(rgba_tensor):
    """
    Convert RGBA tensor to RGB for visualization by applying alpha blending with white background.
    
    Args:
        rgba_tensor: Tensor in [C,H,W] format with C=4 (RGBA)
    
    Returns:
        RGB tensor in [C,H,W] format with C=3
    """
    # Extract RGB and alpha channels
    rgb = rgba_tensor[:3]  # First 3 channels are RGB
    alpha = rgba_tensor[3:4]  # Last channel is alpha
    
    # Create white background
    white_bg = torch.ones_like(rgb)
    
    # Apply alpha blending
    blended = rgb * alpha + white_bg * (1 - alpha)
    
    return blended

def load_and_generate(checkpoint_path, output_path=None, num_layers=None, export_gif=False):
    """
    Load a trained RGBA model and generate a character sequence
    
    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path to save the output image
        num_layers: Number of layers to generate (if None, uses model default)
        export_gif: Whether to export an animated GIF
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
        model_config = checkpoint['model_config']
        if num_layers is None and 'num_layers' in model_config:
            num_layers = model_config['num_layers']
    
    # Default to 18 layers if not specified
    if num_layers is None:
        num_layers = 18
    
    # Initialize model
    model = AutoregressiveRGBAGenerator(num_layers=num_layers).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Try to load a seed image from dataset
    try:
        data = FastRgbaPtDataset("data/full_dataset_rgba.pt")
        seed_image = data.all_data[5][0]
        
        #pdb.set_trace()
        # Make sure the seed is in [C,H,W] format
        if seed_image.dim() == 3 and seed_image.shape[2] == 4:  # [H,W,C] format
            seed_image = seed_image.permute(2, 0, 1)  # Convert to [C,H,W]
        elif seed_image.dim() == 2:  # Grayscale with no channel dimension
            seed_image = seed_image.unsqueeze(0).repeat(4, 1, 1)  # Convert to [C,H,W] with C=4
        
        print(f"Using seed image with shape {seed_image.shape}")
    except Exception as e:
        print(f"Failed to load seed image from dataset: {e}")
        print("Creating blank seed image")
        seed_image = torch.zeros(4, 64, 64)  # Create blank RGBA image
    
    # Generate sequence
    print("Generating RGBA character sequence...")
    model.eval()
    with torch.no_grad():
        sequence = model.generate_sequence(num_layers=num_layers, seed_image=seed_image).cpu()
    
    print(f"Generated sequence with {sequence.shape[0]} layers")
    
    # Create output directory if needed
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Display and save all layers
    plt.figure(figsize=(16, 8))
    
    # Determine grid size
    cols = min(6, sequence.shape[0])
    rows = (sequence.shape[0] + cols - 1) // cols
    
    # Create separate figures for RGB view and alpha channel
    fig_rgb = plt.figure(figsize=(16, 8))
    fig_alpha = plt.figure(figsize=(16, 8))
    
    for i in range(sequence.shape[0]):
        # RGB visualization (alpha-blended with white background)
        plt.figure(fig_rgb.number)
        plt.subplot(rows, cols, i + 1)
        rgb_image = rgba_to_rgb(sequence[i]).numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
        plt.imshow(rgb_image)
        plt.title(f'Layer {i+1} (RGB)')
        plt.axis('off')
        
        # Alpha channel visualization
        plt.figure(fig_alpha.number)
        plt.subplot(rows, cols, i + 1)
        alpha_image = sequence[i, 3].numpy()  # Get alpha channel
        plt.imshow(alpha_image, cmap='gray')
        plt.title(f'Layer {i+1} (Alpha)')
        plt.axis('off')
    
    plt.figure(fig_rgb.number)
    plt.tight_layout()
    plt.figure(fig_alpha.number)
    plt.tight_layout()
    
    # Save the figures if output path provided
    if output_path:
        rgb_path = output_path.rsplit('.', 1)[0] + '_rgb.png'
        alpha_path = output_path.rsplit('.', 1)[0] + '_alpha.png'
        
        plt.figure(fig_rgb.number)
        plt.savefig(rgb_path)
        
        plt.figure(fig_alpha.number)
        plt.savefig(alpha_path)
        
        print(f"Saved RGB visualization to {rgb_path}")
        print(f"Saved alpha visualization to {alpha_path}")
    else:
        plt.show()
    
    # Create a composite image (all layers combined with alpha)
    # For composite, we'll alpha blend each layer onto the previous layers
    composite = torch.zeros(4, sequence.shape[2], sequence.shape[3])
    
    for i in range(sequence.shape[0]):
        current = sequence[i]
        current_alpha = current[3:4]
        # Alpha composite: new = current * alpha + composite * (1 - alpha)
        composite = current * current_alpha + composite * (1 - current_alpha)
    
    # Create composite visualization (RGB and alpha separately)
    fig_composite = plt.figure(figsize=(12, 6))
    
    # RGB composite (alpha-blended with white)
    plt.subplot(1, 2, 1)
    composite_rgb = rgba_to_rgb(composite).numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
    plt.imshow(composite_rgb)
    plt.title('Character Composite (RGB)')
    plt.axis('off')
    
    # Alpha composite
    plt.subplot(1, 2, 2)
    plt.imshow(composite[3].numpy(), cmap='gray')
    plt.title('Character Composite (Alpha)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save composite
    if output_path:
        composite_path = output_path.rsplit('.', 1)[0] + '_composite.png'
        plt.savefig(composite_path)
        print(f"Saved composite image to {composite_path}")
    else:
        plt.show()
    
    # Create and save animated GIF if requested
    if export_gif and output_path:
        gif_path = output_path.rsplit('.', 1)[0] + '.gif'
        frames = []
        
        # Create frames for GIF showing RGB blended with white background
        for i in range(sequence.shape[0]):
            # Convert RGBA to RGB (alpha blended with white)
            rgb_img = rgba_to_rgb(sequence[i]).numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
            
            # Ensure values are in [0, 1] range
            rgb_img = np.clip(rgb_img, 0, 1)
            
            # Create a figure for this frame
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(rgb_img)
            plt.title(f'Layer {i+1}')
            plt.axis('off')
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf)
            frames.append(frame)
            plt.close(fig)
        
        # Save GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=200,  # milliseconds per frame
            loop=0  # loop forever
        )
        print(f"Saved animation to {gif_path}")
        
        # Also create a version showing alpha channel
        alpha_gif_path = output_path.rsplit('.', 1)[0] + '_alpha.gif'
        alpha_frames = []
        
        for i in range(sequence.shape[0]):
            alpha_img = sequence[i, 3].numpy()  # Get alpha channel
            
            # Create a figure for this frame
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(alpha_img, cmap='gray')
            plt.title(f'Layer {i+1} Alpha')
            plt.axis('off')
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = Image.open(buf)
            alpha_frames.append(frame)
            plt.close(fig)
        
        # Save alpha GIF
        alpha_frames[0].save(
            alpha_gif_path,
            save_all=True,
            append_images=alpha_frames[1:],
            duration=200,
            loop=0
        )
        print(f"Saved alpha channel animation to {alpha_gif_path}")
    
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGBA character from trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="generated_rgba_character.png", help="Output image path")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers to generate")
    parser.add_argument("--gif", action="store_true", help="Export animated GIF")
    
    args = parser.parse_args()
    
    load_and_generate(args.checkpoint, args.output, args.layers, args.gif)

# To use this script, run:
# python scripts/rgba_generate.py --checkpoint rgba_results/run_TIMESTAMP/checkpoint_epoch_50.pt --output rgba_character.png --gif