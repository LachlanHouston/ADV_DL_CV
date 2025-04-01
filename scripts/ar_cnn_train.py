import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
from datetime import datetime


# Debugging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Import our model and dataset
from src.models.simple_ar_cnn_model import AutoregressiveCharacterGenerator
from src.data.simple_grayscale_character_dataset import GrayscaleLayerDataset, get_grayscale_dataloaders

def train_model(h5_path, output_dir, batch_size=32, num_epochs=50, lr=0.001):
    """
    Train the autoregressive character generator model.
    
    Args:
        h5_path: Path to the HDF5 dataset
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
    
    # Get dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_grayscale_dataloaders(
        h5_path, 
        target_size=(64, 64),
        batch_size=batch_size,
        max_samples=None  # Set to a small number for quick testing
    )
    print(f"Dataset loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = AutoregressiveCharacterGenerator(num_layers=18).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
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
            loss = criterion(outputs, targets)
            
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
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Save the first batch for visualization
                if epoch % 5 == 0 and val_loss == loss.item():
                    example_inputs = inputs
                    example_targets = targets
                    example_outputs = outputs
        
        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
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
            visualize_results(example_inputs, example_targets, example_outputs, 
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
    generate_full_sequence(model, device, os.path.join(output_dir, 'full_sequence.png'))
    
    print(f"Training completed. Results saved to {output_dir}")
    return model

def visualize_results(inputs, targets, outputs, save_path):
    """
    Visualize model results.
    
    Args:
        inputs: Input images
        targets: Target images
        outputs: Predicted images
        save_path: Path to save visualization
    """
    # Take only the first 4 examples
    num_examples = min(4, inputs.shape[0])
    
    fig, axes = plt.subplots(3, num_examples, figsize=(3*num_examples, 9))
    
    for i in range(num_examples):
        # Input image
        axes[0, i].imshow(inputs[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Input')
        axes[0, i].axis('off')
        
        # Target image
        axes[1, i].imshow(targets[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title('Target')
        axes[1, i].axis('off')
        
        # Output image
        axes[2, i].imshow(outputs[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_full_sequence(model, device, save_path):
    """
    Generate and visualize a full character sequence.
    
    Args:
        model: Trained model
        device: Computation device
        save_path: Path to save visualization
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
                axes[i, j].imshow(sequence[idx, 0].numpy(), cmap='gray')
                axes[i, j].set_title(f'Layer {idx+1}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Character Generator')
    parser.add_argument('--data', type=str, default="data/kenney_dataset_10.h5", help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='./results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"run_{timestamp}")
    
    # Train the model
    model = train_model(
        args.data,
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr
    )