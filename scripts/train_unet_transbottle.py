"""
Main training script for the Hybrid Layer Generator model.
Imports the model architecture from 'model.py' and visualization
utilities from 'visualization.py'.

This script handles:
- Hyperparameter definition
- Dataset loading
- Training loop setup (optimizer, scheduler, loss)
- Epoch iteration with training steps
- Periodic visualization and checkpoint saving
- Final model saving
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F # Keep if used directly, e.g., for F.interpolate in loss check
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import torchvision.transforms as transforms # Keep if needed for LayerPtDataset preprocessing
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import LayerPtDataset # Ensure this returns (input, target, layer_idx)
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss
from src.models.model_unet_transbottle import LayerGeneratorHybrid, VGGPerceptualLoss
from src.utils.visualization import save_image_grid, save_rollout_grid


# --- Hyperparameters ---
subset_fraction = 1.0     # Use 1.0 for full dataset
greyscale = False         # Set to True for grayscale (1 channel), False for color (e.g., 4 channels)
img_size = 64             # Input/Output image size
batch_size = 32           # Adjust based on GPU memory
num_epochs = 200          # Number of training epochs
learning_rate = 1e-4      # Initial learning rate
cnn_start_filters = 32    # Number of filters in the first CNN layer
transformer_embed_dim = 256 # Embedding dimension in the Transformer bottleneck
cnn_depth = 4             # Number of down/up sampling stages in CNN U-Net part
transformer_layers = 6    # Number of layers in the Transformer bottleneck
transformer_heads = 8     # Number of attention heads in the Transformer
weight_decay = 0.05       # Weight decay for AdamW optimizer
use_diff_as_target = True # <<<<<<< ADDED THIS FLAG

# Loss Function Weights (IMPORTANT: TUNE THESE)
lambda_mse = 1.0          # Weight for WeightedProportionalMSELoss
lambda_perceptual = 0.003   # Weight for VGGPerceptualLoss

VISUALIZE_AND_CHECKPOINT_FREQUENCY = 10     # Save images and model every N epochs
SAVE_MODEL = True                          # Set to True to save checkpoints and final model
visualize_rollouts = True                  # Set to True to generate rollout visualizations


# --- Configuration ---
# Update this path to your actual dataset file
data_file = 'data/full_dataset_color.pt'
# Create a unique directory for each run based on timestamp
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = f'results/diff_small_tanh{run_timestamp}/'
# Note: Checkpoints are now saved within the training loop with epoch number

# --- Training Function ---
def train_model(greyscale=True, subset_fraction=1.0):
    """ Trains the LayerGeneratorHybrid model """
    in_chans = 1 if greyscale else 4 # Input channels (1 for grayscale, 4 for RGBA assumed)
    out_chans = in_chans # Output channels usually match input

    print(f"--- Training Configuration ---")
    print(f"Timestamp: {run_timestamp}")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Subset: {subset_fraction*100:.1f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"CNN Filters: {cnn_start_filters}, CNN Depth: {cnn_depth}")
    print(f"Transformer Dim: {transformer_embed_dim}, Layers: {transformer_layers}, Heads: {transformer_heads}")
    print(f"Loss Weights: MSE={lambda_mse}, Perceptual={lambda_perceptual}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset from: {data_file}")
    try:
        # Pass any necessary transforms here if needed by LayerPtDataset
        dataset = LayerPtDataset(data_file, img_size=img_size, transform=None,
                                 greyscale=greyscale, subset_fraction=subset_fraction)
        print(f"Successfully loaded dataset. Number of samples: {len(dataset)}")

        # Determine max layers for embedding (important for the model)
        if hasattr(dataset, 'num_layers') and dataset.num_layers is not None:
             num_layers_max = dataset.num_layers
             print(f"Using num_layers_max = {num_layers_max} from dataset.")
        else:
             # Infer from data or set a default if attribute missing/None
             # Example: try to find max layer_idx in a subset (can be slow)
             # Or set a reasonable default if known
             num_layers_max = 18 # Default value if not found in dataset
             print(f"Warning: Dataset does not provide 'num_layers'. Using default max_layers={num_layers_max} for embedding.")

    except FileNotFoundError:
         print(f"Error: Dataset file not found at {data_file}. Please check the path.")
         return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: Loaded dataset is empty. Cannot train.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=min(4, os.cpu_count() // 2), # Adjust num_workers based on system
                            pin_memory=True, drop_last=True)
    print(f"Dataloader created. Batches per epoch: {len(dataloader)}")
    if len(dataloader) == 0:
        print("Warning: Dataloader has zero batches. Check batch_size vs dataset size.")
        # Optionally exit if no batches can be formed
        # return


    # **** Instantiate the Hybrid Model (imported from model.py) ****
    model = LayerGeneratorHybrid(
        img_size=img_size, in_chans=in_chans, out_chans=out_chans,
        cnn_start_filters=cnn_start_filters, cnn_depth=cnn_depth,
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_layers_max=num_layers_max # Use determined max layers
    )

    # Setup device, model, optimizer, losses, scheduler, scaler
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # elif torch.backends.mps.is_available(): # MPS support can be less stable
    #     device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Print model parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params / 1e6:.2f} M")
    except Exception as e:
        print(f"Could not calculate model parameters: {e}")


    # --- Loss Functions ---
    # Weighted MSE Loss (ensure it's imported correctly)
    if use_diff_as_target:
        criterion_mse = nn.MSELoss()
    else:
        criterion_mse = WeightedProportionalMSELoss(threshold=0.05).to(device)


    # Perceptual Loss (imported from model.py)
    criterion_perceptual = VGGPerceptualLoss(feature_layers=[2, 7, 16, 25, 34]).to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Ensure total_steps is at least 1 for scheduler
    total_steps = max(1, num_epochs * len(dataloader))
    # Cosine annealing scheduler: warms down LR towards eta_min
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    # Automatic Mixed Precision (AMP) scaler for potential speedup on CUDA
    # Enable AMP only if device is CUDA
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")


    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss_total = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_perc = 0.0
        # Progress bar for batches
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # Ensure data unpacking matches dataset __getitem__
                # Expecting: (input_image [B,C,H,W], target_image [B,C,H,W], layer_indices [B])
                input_image, target, layer_idx = batch_data
            except ValueError as e:
                print(f"\nError unpacking batch data at index {batch_idx}. Expected 3 items (input, target, layer_idx). Got: {len(batch_data)}. Error: {e}")
                print("Check your LayerPtDataset __getitem__ method.")
                continue # Skip this batch
            except Exception as e:
                 print(f"\nUnexpected error unpacking batch {batch_idx}: {e}")
                 continue # Skip this batch

            if use_diff_as_target:
                target = target - input_image

            # Move data to the training device
            try:
                input_image = input_image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                layer_idx = layer_idx.to(device, non_blocking=True)
            except Exception as e:
                print(f"\nError moving batch {batch_idx} to device {device}: {e}")
                continue # Skip batch if data transfer fails

            # Reset gradients before forward pass
            optimizer.zero_grad(set_to_none=True) # More memory efficient

            # --- Forward pass with AMP context ---
            with torch.cuda.amp.autocast(enabled=use_amp):
                try:
                    output = model(input_image, layer_idx)

                    # --- Shape Check and Potential Resize ---
                    if output.shape != target.shape:
                        # Check if only spatial dimensions differ (H, W)
                        if output.shape[0:2] == target.shape[0:2] and len(output.shape) == 4 and len(target.shape) == 4:
                            print(f"\nWarning: Resizing model output {output.shape} to match target {target.shape} at batch {batch_idx}.")
                            output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
                        else:
                            # If batch size or channels differ, it's a critical error
                            print(f"\nError: Unrecoverable shape mismatch! Output: {output.shape}, Target: {target.shape}. Skipping batch {batch_idx}.")
                            continue # Skip this batch

                    # --- Calculate Loss Components ---
                    # Ensure input_image is also passed if needed by the loss function (e.g., WeightedProportionalMSELoss)
                    if use_diff_as_target:
                        # nn.MSELoss expects (prediction, target)
                        loss_mse = criterion_mse(output, target)
                    else:
                        # WeightedProportionalMSELoss expects (input, prediction, target)
                        loss_mse = criterion_mse(input_image, output, target)
                        
                    loss_perceptual = criterion_perceptual(output, target) # Compare generated vs target

                    # --- Combine Losses with Weights ---
                    loss = (lambda_mse * loss_mse) + (lambda_perceptual * loss_perceptual)

                except Exception as forward_err:
                    print(f"\nError during forward pass or loss calculation at epoch {epoch+1}, batch {batch_idx}: {forward_err}")
                    # Optionally add more details, e.g., input/output shapes if helpful
                    continue # Skip this batch


            # --- Backward pass and Optimization ---
            # Check for NaN/Inf loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN or Inf loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping backward/step.")
                # Optionally log which component was NaN/Inf
                if torch.isnan(loss_mse) or torch.isinf(loss_mse): print(f"  MSE Loss was: {loss_mse.item()}")
                if torch.isnan(loss_perceptual) or torch.isinf(loss_perceptual): print(f"  Perceptual Loss was: {loss_perceptual.item()}")
                # Reset gradients and skip optimizer step for this batch
                optimizer.zero_grad(set_to_none=True)
                continue # Skip optimization for this batch

            try:
                # Scale the loss for AMP
                scaler.scale(loss).backward()
                # Optional: Gradient clipping (can help stability)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Unscale gradients and step optimizer
                scaler.step(optimizer)
                # Update the scaler for next iteration
                scaler.update()
                # Step the learning rate scheduler (per step/batch)
                scheduler.step()
            except Exception as backward_err:
                 print(f"\nError during backward pass or optimizer step at epoch {epoch+1}, batch {batch_idx}: {backward_err}")
                 # Consider resetting gradients here too
                 optimizer.zero_grad(set_to_none=True)
                 continue # Skip this batch update

            # --- Logging ---
            # Accumulate losses for epoch average (use .item() to get scalar Python number)
            epoch_loss_total += loss.item()
            epoch_loss_mse += loss_mse.item()
            epoch_loss_perc += loss_perceptual.item()
            # Update progress bar postfix with current batch losses and LR
            current_lr = scheduler.get_last_lr()[0] # Get current LR
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{loss_mse.item():.4f}",
                perc=f"{loss_perceptual.item():.4f}",
                lr=f"{current_lr:.1e}"
            )
            # --- End of Batch Loop ---

        # --- End of Epoch ---
        progress_bar.close() # Close the tqdm bar for the epoch
        num_batches = len(dataloader)
        if num_batches > 0:
            # Calculate average losses for the epoch
            avg_epoch_loss = epoch_loss_total / num_batches
            avg_mse_loss_comp = (epoch_loss_mse / num_batches) # Component value
            avg_perc_loss_comp = (epoch_loss_perc / num_batches) # Component value
            # Weighted contributions to total loss
            avg_mse_loss_weighted = lambda_mse * avg_mse_loss_comp
            avg_perc_loss_weighted = lambda_perceptual * avg_perc_loss_comp

            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} Summary -> Avg Loss: {avg_epoch_loss:.4f} "
                  f"(MSE: {avg_mse_loss_weighted:.4f}, Perc: {avg_perc_loss_weighted:.4f}) - LR: {current_lr:.2e}")
        else:
            # Handle case where dataloader might be empty or all batches were skipped
            print(f"Epoch {epoch+1}/{num_epochs} - No batches successfully processed.")
            avg_epoch_loss = float('inf') # Set placeholder loss if no batches ran

        # --- Visualization and Checkpointing ---
        # Perform at specified frequency OR on the first epoch (epoch==0) OR on the last epoch
        if (epoch + 1) % VISUALIZE_AND_CHECKPOINT_FREQUENCY == 0 or epoch == num_epochs - 1 or epoch == 0:
            print(f"--- Running Visualization & Checkpointing for Epoch {epoch+1} ---")
            try:
                # Use last valid batch data for visualization
                # Check if variables from the loop exist and are tensors
                if ('input_image' in locals() and isinstance(input_image, torch.Tensor) and
                    'target' in locals() and isinstance(target, torch.Tensor) and
                    'output' in locals() and isinstance(output, torch.Tensor)):

                    # Set model to evaluation mode for consistent visualization
                    model.eval()
                    with torch.no_grad(): # No gradients needed
                         # Re-run inference in eval mode for potentially cleaner output (BN stats fixed)
                         with torch.cuda.amp.autocast(enabled=use_amp):
                              output_vis = model(input_image, layer_idx) # Use last batch input/layer_idx

                         # Save image grid (imported from visualization.py)
                         save_image_grid(input_image, target, output_vis, epoch, save_dir=save_dir,
                                         use_diff_as_target=use_diff_as_target)
                         print(f"Visualization grid saved for epoch {epoch+1}")

                    # Set model back to training mode
                    model.train()
                else:
                     print("Skipping visualization: Last batch data not available or invalid.")

                # --- Optional roll‑out visualisation ---
                if visualize_rollouts:
                    # Ensure dataset object is available
                    if 'dataset' in locals():
                        # save_rollout_grid imported from visualization.py
                         save_rollout_grid(model, save_dir, img_size, in_chans, device, epoch,
                                         dataset,
                                         n_rollouts=min(5, batch_size), # Limit rollouts if batch size is small
                                         num_layers=num_layers_max, # Use the determined max layers
                                         use_diff_as_target=use_diff_as_target)
                    else:
                        print("Skipping rollouts: Dataset object not available.")


                # --- Save Model Checkpoint ---
                if SAVE_MODEL:
                    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
                    # Save model state, optimizer, scheduler, epoch, loss, and hyperparameters
                    save_data = {
                        'epoch': epoch + 1, # Save next epoch number to resume from
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if use_amp else None,
                        'loss': avg_epoch_loss, # Save average epoch loss
                        'hyperparameters': {
                             'img_size': img_size, 'greyscale': greyscale, 'in_chans': in_chans,
                             'batch_size': batch_size, 'learning_rate': learning_rate,
                             'cnn_start_filters': cnn_start_filters, 'cnn_depth': cnn_depth,
                             'transformer_embed_dim': transformer_embed_dim, 'transformer_layers': transformer_layers,
                             'transformer_heads': transformer_heads, 'num_layers_max': num_layers_max,
                             'lambda_mse': lambda_mse, 'lambda_perceptual': lambda_perceptual,
                             'weight_decay': weight_decay,
                             'use_diff_as_target': use_diff_as_target # <<<<< Save the flag
                        }
                    }
                    torch.save(save_data, checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")

            except Exception as e:
                import traceback
                print(f"\nError during visualization/checkpointing at epoch {epoch+1}: {e}")
                print("Traceback:")
                traceback.print_exc()
                # Ensure model is back in train mode even if an error occurs
                model.train()
            print("-" * 20) # Separator after checkpointing info
            # --- End Checkpointing Block ---

    # --- End of Training Loop ---
    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print("-" * 50)
    print(f"Training completed in {total_training_time / 60:.2f} minutes ({total_training_time:.1f} seconds).")

    # Save the final model state dictionary (only the model weights)
    if SAVE_MODEL:
        final_model_path = os.path.join(save_dir, 'hybrid_layer_generator_final.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state_dict saved as {final_model_path}")

    print(f"Results (checkpoints, visualizations) saved in: {save_dir}")
    print("-" * 50)



if __name__ == '__main__':
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    print("Training completed successfully.")