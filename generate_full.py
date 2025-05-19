import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from PIL import Image
import io

from src.data.simple_grayscale_character_dataset import FastPtDataset
import pdb

# Import the model
from src.models.simple_ar_cnn_model import AutoregressiveCharacterGenerator

def load_and_generate(checkpoint_path, output_path=None, num_layers=None, export_gif=False):
    """
    Load a trained model and generate a character sequence
    
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
    model = AutoregressiveCharacterGenerator(num_layers=num_layers).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    

    data = FastPtDataset("data/full_dataset.pt")
    seed_image = data.all_data[0][0]
    seed_image = seed_image.unsqueeze(0)
    # Generate sequence
    print("Generating character sequence...")
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
    
    for i in range(sequence.shape[0]):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(sequence[i, 0].numpy(), cmap='gray')
        plt.title(f'Layer {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    # Also create a composite image (sum of all layers)
    composite = torch.sum(sequence, dim=0)[0].numpy()
    # Normalize composite to [0, 1]
    composite = np.clip(composite / composite.max(), 0, 1)
    
    # Create composite visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(composite, cmap='gray')
    plt.title('Character (All Layers Combined)')
    plt.axis('off')
    
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
        
        # Create frames for GIF
        for i in range(sequence.shape[0]):
            img = sequence[i, 0].numpy()
            
            # Create a figure for this frame
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
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
    
    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate character from trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="generated_character.png", help="Output image path")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers to generate")
    parser.add_argument("--gif", action="store_true", help="Export animated GIF")
    
    args = parser.parse_args()
    
    load_and_generate(args.checkpoint, args.output, args.layers, args.gif)

# To use this script, run:
# python quick_generate.py --checkpoint results/latest/checkpoint_epoch_50.pt --output character.png --gif