import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import time

# Debugging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our model and dataset
from src.models.rgba_ar_cnn_model import AutoregressiveRGBAGenerator
from src.data.rgba_character_dataset import get_rgba_pt_dataloaders
from src.utils.rgba_weighted_mse_loss import RGBAWeightedMSELoss, RGBAWeightedProportionalMSELoss


def train_model(dataset_path, metadata_path, output_dir, batch_size=32, num_epochs=50, lr=0.001):
    """
    Train the autoregressive RGBA character generator model with pre-processed PyTorch data.
    
    Args:
        dataset_path: Path to the PyTorch dataset file
        metadata_path: Path to the metadata file
        output_dir: Directory to save outputs
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        lr: Learning rate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders with fast loading
    print("Loading RGBA dataset...")
    data_loading_start = time.time()
    train_loader, val_loader = get_rgba_pt_dataloaders(
        dataset_path,
        metadata_path,
        batch_size=batch_size
    )
    data_loading_time = time.time() - data_loading_start
    print(f"Dataset loaded in {data_loading_time:.2f}s")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Load metadata to determine model params
    metadata = torch.load(metadata_path) if metadata_path else None
    num_layers = metadata['num_layers'] if metadata else 18
    
    # Initialize model
    model = AutoregressiveRGBAGenerator(num_layers=num_layers).to(device)
    
    # Loss function and optimizer
    # criterion = RGBAWeightedMSELoss(threshold=0.05, alpha=10.0, alpha_channel_weight=2.0)
    criterion = RGBAWeightedProportionalMSELoss(threshold=0.05, alpha_channel_weight=2.0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    total_train_time = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets, layer_idxs in tqdm(train_loader, desc="Training"):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            layer_idxs = layer_idxs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, layer_idxs)
            
            # Calculate loss
            loss = criterion(inputs, outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, layer_idxs in tqdm(val_loader, desc="Validating"):
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                layer_idxs = layer_idxs.to(device)
                
                # Forward pass
                outputs = model(inputs, layer_idxs)
                
                # Calculate loss
                loss = criterion(inputs, outputs, targets)
                val_loss += loss.item()
                
                # Save the first batch for visualization
                if epoch % 5 == 0 and val_loss == loss.item():
                    example_inputs = inputs
                    example_targets = targets
                    example_outputs = outputs
        
        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Epoch time: {epoch_time:.2f}s")
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Visualize results every 5 epochs
        if epoch % 5 == 0:
            visualize_rgba_results(example_inputs, example_targets, example_outputs, 
                             os.path.join(output_dir, f'results_epoch_{epoch+1}.png'))
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    # Generate and save a full sequence
    generate_full_rgba_sequence(model, device, os.path.join(output_dir, 'full_sequence.png'))
    
    avg_epoch_time = total_train_time / num_epochs
    print(f"Training completed. Results saved to {output_dir}")
    print(f"Total training time: {total_train_time:.2f}s, Average epoch time: {avg_epoch_time:.2f}s")
    
    return model

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

def visualize_rgba_results(inputs, targets, outputs, save_path):
    """
    Visualize model results for RGBA images.
    
    Args:
        inputs: Input tensors [B,C,H,W] with C=4
        targets: Target tensors [B,C,H,W] with C=4
        outputs: Output tensors [B,C,H,W] with C=4
        save_path: Path to save the visualizations
    """
    # Take only the first 4 examples
    num_examples = min(4, inputs.shape[0])
    
    for i in range(num_examples):
        # Convert RGBA to RGB for visualization
        # All tensors are in [C,H,W] format with RGBA channels
        input_rgb = rgba_to_rgb(inputs[i]).cpu().numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C] for matplotlib
        target_rgb = rgba_to_rgb(targets[i]).cpu().numpy().transpose(1, 2, 0)
        output_rgb = rgba_to_rgb(outputs[i]).cpu().numpy().transpose(1, 2, 0)
        
        # Also visualize the alpha channel separately
        input_alpha = inputs[i, 3].cpu().numpy()  # Alpha is the 4th channel (index 3)
        target_alpha = targets[i, 3].cpu().numpy()
        output_alpha = outputs[i, 3].cpu().numpy()
        
        # Create subplots with RGB on left and alpha on right
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        
        # Input image
        axes[0, 0].imshow(input_rgb)
        axes[0, 0].set_title('Input RGB')
        axes[0, 0].axis('off')
        axes[0, 1].imshow(input_alpha, cmap='gray')
        axes[0, 1].set_title('Input Alpha')
        axes[0, 1].axis('off')
        
        # Target image
        axes[1, 0].imshow(target_rgb)
        axes[1, 0].set_title('Target RGB')
        axes[1, 0].axis('off')
        axes[1, 1].imshow(target_alpha, cmap='gray')
        axes[1, 1].set_title('Target Alpha')
        axes[1, 1].axis('off')
        
        # Output image
        axes[2, 0].imshow(output_rgb)
        axes[2, 0].set_title('Predicted RGB')
        axes[2, 0].axis('off')
        axes[2, 1].imshow(output_alpha, cmap='gray')
        axes[2, 1].set_title('Predicted Alpha')
        axes[2, 1].axis('off')
        
        # Save each example as a separate file
        example_path = save_path.replace('.png', f'_example_{i}.png')
        plt.tight_layout()
        plt.savefig(example_path)
        plt.close()
    
    # Create a summary image with all examples
    fig, axes = plt.subplots(3, num_examples, figsize=(4*num_examples, 12))
    
    for i in range(num_examples):
        input_rgb = rgba_to_rgb(inputs[i]).cpu().numpy().transpose(1, 2, 0)
        target_rgb = rgba_to_rgb(targets[i]).cpu().numpy().transpose(1, 2, 0)
        output_rgb = rgba_to_rgb(outputs[i]).cpu().numpy().transpose(1, 2, 0)
        
        axes[0, i].imshow(input_rgb)
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_rgb)
        axes[1, i].set_title('Target')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(output_rgb)
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_full_rgba_sequence(model, device, save_path):
    """
    Generate and visualize a full RGBA character sequence.
    
    Args:
        model: The trained model
        device: Device to use for generation
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Generate a full sequence
    with torch.no_grad():
        sequence = model.generate_sequence().cpu()
    
    # Visualize the sequence
    num_layers = sequence.shape[0]
    cols = 6
    rows = (num_layers + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_layers:
                # Convert RGBA to RGB for visualization
                # Tensors are in [C,H,W] format
                rgb_image = rgba_to_rgb(sequence[idx]).numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                if rows == 1:
                    axes[j].imshow(rgb_image)
                    axes[j].set_title(f'Layer {idx+1}')
                    axes[j].axis('off')
                elif cols == 1:
                    axes[i].imshow(rgb_image)
                    axes[i].set_title(f'Layer {idx+1}')
                    axes[i].axis('off')
                else:
                    axes[i, j].imshow(rgb_image)
                    axes[i, j].set_title(f'Layer {idx+1}')
                    axes[i, j].axis('off')
            elif rows == 1:
                axes[j].axis('off')
            elif cols == 1:
                axes[i].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Also save a visualization showing alpha channel separately
    fig, axes = plt.subplots(rows, cols*2, figsize=(3*cols*2, 3*rows))
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < num_layers:
                # RGB view
                rgb_image = rgba_to_rgb(sequence[idx]).numpy().transpose(1, 2, 0)  # [C,H,W] -> [H,W,C]
                
                # Alpha channel view
                alpha_image = sequence[idx, 3].numpy()
                
                if rows == 1:
                    axes[j*2].imshow(rgb_image)
                    axes[j*2].set_title(f'Layer {idx+1} RGB')
                    axes[j*2].axis('off')
                    
                    axes[j*2+1].imshow(alpha_image, cmap='gray')
                    axes[j*2+1].set_title(f'Layer {idx+1} Alpha')
                    axes[j*2+1].axis('off')
                else:
                    axes[i, j*2].imshow(rgb_image)
                    axes[i, j*2].set_title(f'Layer {idx+1} RGB')
                    axes[i, j*2].axis('off')
                    
                    axes[i, j*2+1].imshow(alpha_image, cmap='gray')
                    axes[i, j*2+1].set_title(f'Layer {idx+1} Alpha')
                    axes[i, j*2+1].axis('off')
            elif rows == 1:
                axes[j*2].axis('off')
                axes[j*2+1].axis('off')
            else:
                axes[i, j*2].axis('off')
                axes[i, j*2+1].axis('off')
    
    plt.tight_layout()
    alpha_save_path = save_path.replace('.png', '_with_alpha.png')
    plt.savefig(alpha_save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RGBA Character Generator with PyTorch dataset')
    parser.add_argument('--dataset', type=str, default="data/full_dataset_rgba.pt", help='Path to PyTorch RGBA dataset file')
    parser.add_argument('--metadata', type=str, default=None, help='Path to metadata file')
    parser.add_argument('--output', type=str, default='./rgba_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    
    # Train the model
    model = train_model(
        args.dataset,
        args.metadata,
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )