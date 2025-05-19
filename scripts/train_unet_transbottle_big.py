import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Ensure project structure allows imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import necessary modules from project ---
from src.data import LayerPtDataset
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss
from src.models.model_unet_transbottle import LayerGeneratorHybrid, VGGPerceptualLoss
from src.utils.visualization import save_image_grid, save_rollout_grid

run_name = "big_diff_split_mse"

# --- Hyperparameters ---
subset_fraction = 1.0     # Fraction of the *original* dataset file to load initially
greyscale = False         # Set to True for grayscale (1 channel), False for color (e.g., 4 channels)
img_size = 128            # Input/Output image size
batch_size = 32           # Adjust based on GPU memory
num_epochs = 200          # Number of training epochs
learning_rate = 1e-4      # Initial learning rate
cnn_start_filters = 64    # Number of filters in the first CNN layer
transformer_embed_dim = 512 # Embedding dimension in the Transformer bottleneck
cnn_depth = 4             # Number of down/up sampling stages in CNN U-Net part
transformer_layers = 8    # Number of layers in the Transformer bottleneck
transformer_heads = 8     # Number of attention heads
weight_decay = 0.05       # Weight decay for AdamW optimizer

# Loss Function Weights
lambda_mse = 1.0
lambda_perceptual = 0.005
use_diff_as_target = True # Calculate loss on (output) vs (target - input)

# Split Proportions
train_split = 0.8
val_split = 0.1
test_split = 0.1
if not np.isclose(train_split + val_split + test_split, 1.0):
    raise ValueError("Split proportions must sum to 1.0")

# Visualization and Checkpointing
VISUALIZE_FREQUENCY = 10 # Visualize validation examples every N epochs
CHECKPOINT_FREQUENCY = 1000 # Save standard checkpoints every N epochs (optional, less critical now)
SAVE_BEST_MODEL = True   # Save the model with the best validation loss
SAVE_MODEL = True         # General flag to enable saving
visualize_rollouts = True # Generate rollout visualizations periodically

# --- Configuration ---
data_file = 'data/full_dataset_128.pt'
run_timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir = f'results/{run_timestamp}_{run_name}/'
best_model_filename = 'best_model_val.pth' # Filename for the best model based on validation
plot_filename = 'loss_curves.png'
final_model_filename = 'final_model_state.pth' # Optional: save final state regardless of performance

# --- Evaluation Function ---
def evaluate(model, dataloader, criterion_mse, criterion_perceptual, device, use_amp, lambda_mse, lambda_perceptual, use_diff_as_target, desc="Evaluating"):
    """Evaluates the model on a given dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    total_mse_loss = 0.0
    total_perc_loss = 0.0
    num_batches = len(dataloader)

    if num_batches == 0:
        print(f"Warning: Dataloader for '{desc}' is empty.")
        return 0.0, 0.0, 0.0 # Return zero losses if no data

    with torch.no_grad(): # Disable gradient calculations
        progress_bar = tqdm(dataloader, desc=desc, leave=False)
        for batch_data in progress_bar:
            input_image, target, layer_idx = batch_data

            original_target = target # Keep original target
            if use_diff_as_target:
                target = target - input_image # Calculate difference target

            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)
            original_target = original_target.to(device, non_blocking=True) # Also move original


            with torch.cuda.amp.autocast(enabled=use_amp): # Deprecation warning exists here, fix below
                output = model(input_image, layer_idx)

                # Ensure shapes match for loss calculation
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

                # --- Calculate Loss Components ---
                # FIX: Use conditional logic for criterion_mse call
                if use_diff_as_target:
                    # In diff mode, criterion_mse is nn.MSELoss, expects (prediction, target)
                    # Here 'output' is predicted diff, 'target' is actual diff.
                    loss_mse = criterion_mse(output, target)
                else:
                    # In non-diff mode, criterion_mse is WeightedProportionalMSELoss, expects (input, prediction, target)
                    # Here 'output' is predicted target, 'target' is actual target.
                    loss_mse = criterion_mse(input_image, output, target) # This call is now correct for this case

                # Perceptual loss compares the generated *image* vs original target
                if use_diff_as_target:
                    output_image = output + input_image # Reconstruct the full image prediction
                    loss_perceptual = criterion_perceptual(output_image, original_target)
                else:
                    # If not using diff target, output IS the image prediction
                    loss_perceptual = criterion_perceptual(output, original_target)


                loss = (lambda_mse * loss_mse) + (lambda_perceptual * loss_perceptual)  

            # Simplified: Assume loss is valid (no NaN/Inf check)
            total_loss += loss.item()
            total_mse_loss += loss_mse.item() # Accumulate the component loss
            total_perc_loss += loss_perceptual.item() # Accumulate the component loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_mse = total_mse_loss / num_batches # Calculate average component loss
    avg_perc = total_perc_loss / num_batches # Calculate average component loss
    return avg_loss, avg_mse, avg_perc


# --- Training Function ---
def train_model(greyscale=True, subset_fraction=1.0):
    """ Trains the LayerGeneratorHybrid model with train/val/test splits. """
    in_chans = 1 if greyscale else 4
    out_chans = in_chans

    print(f"--- Training Configuration ---")
    print(f"Timestamp: {run_timestamp}")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Initial Subset: {subset_fraction*100:.1f}%")
    print(f"Splits: Train={train_split*100:.0f}%, Val={val_split*100:.0f}%, Test={test_split*100:.0f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Use Difference as Target: {use_diff_as_target}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    os.makedirs(save_dir, exist_ok=True)
    plot_save_path = os.path.join(save_dir, plot_filename)
    best_model_path = os.path.join(save_dir, best_model_filename)
    final_model_path = os.path.join(save_dir, final_model_filename)

    # --- Dataset Loading and Splitting (Simplified - No try-except) ---
    print(f"Loading full dataset from: {data_file}")
    # Load the dataset class
    full_dataset = LayerPtDataset(data_file, img_size=img_size, transform=None,
                                  greyscale=greyscale, subset_fraction=subset_fraction)
    print(f"Successfully loaded dataset. Number of transitions: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Error: Dataset contains no transitions. Exiting.")
        return

    # Determine max layers from the dataset instance
    if hasattr(full_dataset, 'num_layers') and full_dataset.num_layers is not None:
         # Max index 't' for input layer t is num_layers - 2 (since we need t+1)
         # Max value for embedding needs to cover up to num_layers-1
         num_layers_max = full_dataset.num_layers
         print(f"Using num_layers_max = {num_layers_max} (from dataset's {full_dataset.num_layers} layers) for embedding.")
    else:
         num_layers_max = 18 # Fallback default
         print(f"Warning: Could not reliably determine num_layers_max from dataset. Using default: {num_layers_max}")

    # --- Split the dataset ---
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size # Ensure all data is used

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}, Test={test_size}")
    if train_size == 0 or val_size == 0:
         print("Error: Train or Validation set size is 0. Cannot proceed.")
         return

    # Use random_split for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # --- Create DataLoaders ---
    num_workers = min(4, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, # No shuffle for val/test
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    if len(train_loader) == 0:
        print("Error: Training dataloader has zero batches. Cannot train.")
        return


    # **** Instantiate the Hybrid Model ****
    model = LayerGeneratorHybrid(
        img_size=img_size, in_chans=in_chans, out_chans=out_chans,
        cnn_start_filters=cnn_start_filters, cnn_depth=cnn_depth,
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_layers_max=num_layers_max
    )

    # Setup device, model, optimizer, losses, scheduler, scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Simplified: Assume parameter counting works
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")

    # --- Loss Functions ---
    if use_diff_as_target:
        criterion_mse = nn.MSELoss()
    else:
        criterion_mse = WeightedProportionalMSELoss(threshold=0.05).to(device)
    
    criterion_perceptual = VGGPerceptualLoss(feature_layers=[2, 7, 16, 25, 34]).to(device)

    # --- Optimizer and Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")


    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    # --- Loss Tracking ---
    train_losses_epoch = []
    val_losses_epoch = []
    best_val_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        epoch_loss_total_train = 0.0
        epoch_loss_mse_train = 0.0
        epoch_loss_perc_train = 0.0
        num_train_batches = len(train_loader)

        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar_train):
            # Simplified: Assume data loading works
            input_image, target, layer_idx = batch_data

            original_target = target # Keep original
            if use_diff_as_target:
                target = target - input_image # Target is the difference

            # Simplified: Assume moving to device works
            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)
            original_target = original_target.to(device, non_blocking=True)


            optimizer.zero_grad(set_to_none=True)


            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input_image, layer_idx)

                if output.shape != target.shape:
                    print(f"\nWarning: Resizing model output {output.shape} to match target {target.shape} in train batch {batch_idx}.")
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

                # --- Calculate Loss Components ---
                # Ensure input_image is also passed if needed by the loss function (e.g., WeightedProportionalMSELoss)
                if use_diff_as_target:
                    # In diff mode, criterion_mse is likely nn.MSELoss, expects (prediction, target)
                    # Here 'output' is predicted diff, 'target' is actual diff.
                    loss_mse = criterion_mse(output, target)
                else:
                    # In non-diff mode, criterion_mse is WeightedProportionalMSELoss, expects (input, prediction, target)
                    # Here 'output' is predicted target, 'target' is actual target.
                    loss_mse = criterion_mse(input_image, output, target)

                    loss_perceptual = criterion_perceptual(output, target) # Compare generated vs target

                if use_diff_as_target:
                    output_image = output + input_image
                    loss_perceptual = criterion_perceptual(output_image, original_target)
                else:
                    loss_perceptual = criterion_perceptual(output, original_target)

                loss = (lambda_mse * loss_mse) + (lambda_perceptual * loss_perceptual)

            # Simplified: Assume loss is valid, backward/step works
            # Check for NaN/Inf loss before backward pass (keeping this check as it's crucial)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf train loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue # Skip optimization for this batch

            scaler.scale(loss).backward()
            # Optional: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step scheduler per batch/step

            # Accumulate training losses
            epoch_loss_total_train += loss.item()
            epoch_loss_mse_train += loss_mse.item()
            epoch_loss_perc_train += loss_perceptual.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar_train.set_postfix(
                loss=f"{loss.item():.4f}", mse=f"{loss_mse.item():.4f}", perc=f"{loss_perceptual.item():.4f}", lr=f"{current_lr:.1e}"
            )

        progress_bar_train.close()
        # Calculate average training loss for the epoch
        # Handle division by zero just in case num_train_batches is 0 (though checked earlier)
        avg_train_loss = epoch_loss_total_train / num_train_batches if num_train_batches > 0 else 0.0
        avg_train_mse = lambda_mse * (epoch_loss_mse_train / num_train_batches) if num_train_batches > 0 else 0.0
        avg_train_perc = lambda_perceptual * (epoch_loss_perc_train / num_train_batches) if num_train_batches > 0 else 0.0
        train_losses_epoch.append(avg_train_loss)


        # --- Validation Phase ---
        avg_val_loss, avg_val_mse_comp, avg_val_perc_comp = evaluate(
            model, val_loader, criterion_mse, criterion_perceptual, device, use_amp,
            lambda_mse, lambda_perceptual, use_diff_as_target, desc="Validating"
        )
        val_losses_epoch.append(avg_val_loss)
        # Calculate weighted validation components for printing
        avg_val_mse_weighted = lambda_mse * avg_val_mse_comp
        avg_val_perc_weighted = lambda_perceptual * avg_val_perc_comp


        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} -> "
              f"Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f}, Perc: {avg_train_perc:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse_weighted:.4f}, Perc: {avg_val_perc_weighted:.4f}) | " # Show weighted components
              f"LR: {current_lr:.2e}")

        # --- Save Best Model ---
        if SAVE_BEST_MODEL and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model state dictionary along with epoch and loss
            save_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'best_val_loss': best_val_loss,
                'hyperparameters': { # Store key hyperparameters
                     'img_size': img_size, 'greyscale': greyscale, 'in_chans': in_chans,
                     'batch_size': batch_size, 'learning_rate': learning_rate,
                     'cnn_start_filters': cnn_start_filters, 'cnn_depth': cnn_depth,
                     'transformer_embed_dim': transformer_embed_dim, 'transformer_layers': transformer_layers,
                     'transformer_heads': transformer_heads, 'num_layers_max': num_layers_max,
                     'lambda_mse': lambda_mse, 'lambda_perceptual': lambda_perceptual,
                     'weight_decay': weight_decay, 'use_diff_as_target': use_diff_as_target
                }
            }
            torch.save(save_data, best_model_path)
            print(f"✅ New best model saved with Val Loss: {best_val_loss:.4f} at {best_model_path}")


        # --- Visualization (Optional - using validation data) ---
        # Simplified: Assume visualization functions work, remove try-except
        if (epoch + 1) % VISUALIZE_FREQUENCY == 0 or epoch == num_epochs - 1:
            print(f"--- Running Visualization for Epoch {epoch+1} ---")
            # Get a fixed batch from validation loader for consistent visualization
            vis_batch = next(iter(val_loader), None)
            if vis_batch:
                vis_input, vis_target, vis_layer_idx = vis_batch
                vis_input = vis_input.to(device)
                vis_target = vis_target.to(device) # This is original target
                vis_layer_idx = vis_layer_idx.to(device)

                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        vis_output_diff = model(vis_input, vis_layer_idx) # Model predicts difference if use_diff_as_target=True

                # Decide what to visualize based on use_diff_as_target
                if use_diff_as_target:
                    target_to_show = vis_target - vis_input # The difference target
                    output_to_show = vis_output_diff       # The predicted difference
                else:
                    target_to_show = vis_target            # The actual target image
                    output_to_show = vis_output_diff       # The predicted image


                save_image_grid(vis_input, target_to_show, output_to_show, epoch + 1, save_dir=save_dir,
                                use_diff_as_target=use_diff_as_target) # Pass flag for correct title/interpretation
                print(f"Visualization grid saved for epoch {epoch+1}")
                model.train() # Set back to train mode
            else:
                print("Skipping visualization: Could not get validation batch.")

            # Optional roll‑out visualisation (using full_dataset reference)
            if visualize_rollouts:
                 save_rollout_grid(model, save_dir, img_size, in_chans, device, epoch + 1,
                                   full_dataset, # Pass the original full dataset reference
                                   n_rollouts=min(5, batch_size),
                                   num_layers=num_layers_max, # Use determined max layers for rollout generation
                                   use_diff_as_target=use_diff_as_target)
                 print(f"Rollout visualization saved for epoch {epoch+1}")


        # --- Optional: Save Regular Checkpoint ---
        if SAVE_MODEL and (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
             checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
             # Save similarly to best model, but maybe without optimizer state if space is a concern
             chkpt_data = {
                 'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'val_loss': avg_val_loss, # Record current val loss
                 'hyperparameters': save_data['hyperparameters'] if 'save_data' in locals() else None # Reuse hypers if available
             }
             torch.save(chkpt_data, checkpoint_path)
             print(f"Checkpoint saved: {checkpoint_path}")
        # --- End Epoch ---

    # --- End of Training Loop ---
    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print("-" * 50)
    print(f"Training completed in {total_training_time / 60:.2f} minutes ({total_training_time:.1f} seconds).")

    # --- Final Test Evaluation ---
    print("\n--- Evaluating Best Model on Test Set ---")
    if os.path.exists(best_model_path):
        # Load the best model state
        checkpoint = torch.load(best_model_path, map_location=device)
        # Need to re-instantiate the model structure before loading state_dict
        best_model = LayerGeneratorHybrid(
            img_size=img_size, in_chans=in_chans, out_chans=out_chans,
            cnn_start_filters=cnn_start_filters, cnn_depth=cnn_depth,
            transformer_embed_dim=transformer_embed_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            num_layers_max=num_layers_max # Ensure this matches saved model
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_model.to(device)
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')} with Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

        # Evaluate on the test set
        test_loss, test_mse_comp, test_perc_comp = evaluate(
            best_model, test_loader, criterion_mse, criterion_perceptual, device, use_amp,
            lambda_mse, lambda_perceptual, use_diff_as_target, desc="Testing"
        )
        test_mse_weighted = lambda_mse * test_mse_comp
        test_perc_weighted = lambda_perceptual * test_perc_comp

        print("-" * 30)
        print(f"Final Test Set Performance (using best model):")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test MSE (weighted): {test_mse_weighted:.4f} (Component: {test_mse_comp:.4f})")
        print(f"  Test Perceptual (weighted): {test_perc_weighted:.4f} (Component: {test_perc_comp:.4f})")
        print("-" * 30)
    else:
        print("Could not find the best model file to evaluate on the test set.")
        test_loss = None # Indicate test loss wasn't calculated

    # --- Plotting Losses ---
    print(f"Generating loss plot at {plot_save_path}")
    plot_start_epoch = 20 # Define the epoch number you want to start plotting from

    # Ensure we have enough epochs to plot from the desired start
    if num_epochs >= plot_start_epoch:
        # Adjust the range for the x-axis (epochs)
        # We want epochs from plot_start_epoch up to num_epochs (inclusive)
        epochs_to_plot = range(plot_start_epoch, num_epochs + 1)

        # Slice the loss lists to get data from the desired start epoch onwards
        # Epoch plot_start_epoch corresponds to index plot_start_epoch - 1
        train_losses_to_plot = train_losses_epoch[plot_start_epoch - 1:]
        val_losses_to_plot = val_losses_epoch[plot_start_epoch - 1:]

        plt.figure(figsize=(10, 6))
        # Use the adjusted epochs and sliced loss data for plotting
        plt.plot(epochs_to_plot, train_losses_to_plot, label=f'Training Loss (Epoch {plot_start_epoch}+)')
        plt.plot(epochs_to_plot, val_losses_to_plot, label=f'Validation Loss (Epoch {plot_start_epoch}+)')
        plt.title(f'Training and Validation Loss (Epochs {plot_start_epoch}-{num_epochs})') # Update title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # Add final test loss as a horizontal line - this part doesn't need slicing
        if test_loss is not None:
             plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Final Test Loss ({test_loss:.4f})')

        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        print(f"Loss plot saved (showing epochs {plot_start_epoch}-{num_epochs}).")
        # plt.show() # Optional: display plot if running interactively

    else:
        # Handle the case where the total number of epochs is less than the desired start epoch
        print(f"Warning: Total epochs ({num_epochs}) is less than the desired plot start epoch ({plot_start_epoch}). Plotting all epochs.")
        # Fallback to plotting all epochs if training was too short
        epochs_all = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_all, train_losses_epoch, label='Training Loss')
        plt.plot(epochs_all, val_losses_epoch, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs (Full Range)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if test_loss is not None:
             plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Final Test Loss ({test_loss:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        print("Loss plot saved (full range shown due to short training).")

    # --- Optional: Save Final Model State ---
    if SAVE_MODEL:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model state_dict saved as {final_model_path}")

    print(f"Results (best model, plot, checkpoints) saved in: {save_dir}")
    print("-" * 50)


if __name__ == '__main__':
    # Simplified: Assume args are correct, remove try-except
    train_model(greyscale=greyscale, subset_fraction=subset_fraction)
    print("Training script finished.")