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
from torchvision.utils import save_image # Added for image generation
import random # Added for image generation

# --- Ensure project structure allows imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import necessary modules from project ---
from src.data import LayerPtDataset
from src.utils.weighted_proportional_mse_loss import WeightedProportionalMSELoss
from src.models.model_unet_transbottle import LayerGeneratorHybrid, VGGPerceptualLoss
from src.utils.visualization import save_image_grid, save_rollout_grid

run_name = "small_noPerc"

USE_BIG_MODEL = False

# --- Hyperparameters ---
if USE_BIG_MODEL:
    # Hyperparameters for train_unet_transbottle_big_save.py (current set)
    generate_image_set = True # If true, generate and save 100 images using the best model
    generated_images_folder_name = 'generated_100'

    num_epochs = 200          # Number of training epochs
    use_diff_as_target = True # Calculate loss on (output) vs (target - input)

    subset_fraction = 1.0     # Fraction of the *original* dataset file to load initially
    greyscale = False         # Set to True for grayscale (1 channel), False for color (e.g., 4 channels)
    img_size = 128            # Input/Output image size
    batch_size = 32           # Adjust based on GPU memory
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

else:
    # Hyperparameters from train_unet_transbottle.py
    generate_image_set = True
    generated_images_folder_name = 'generated_100_small_params'

    num_epochs = 200        
    use_diff_as_target = False

    subset_fraction = 1.0
    greyscale = False         
    img_size = 64             
    batch_size = 32           
    learning_rate = 1e-4      
    cnn_start_filters = 32    
    transformer_embed_dim = 256
    cnn_depth = 4             
    transformer_layers = 6    
    transformer_heads = 8     
    weight_decay = 0.05       

    # Loss Function Weights
    lambda_mse = 1.0          
    lambda_perceptual = 0# 0.003   


# Split Proportions (assuming these are common or specific to big_save structure)
train_split = 0.8
val_split = 0.1
test_split = 0.1
if not np.isclose(train_split + val_split + test_split, 1.0):
    raise ValueError("Split proportions must sum to 1.0")

# Visualization and Checkpointing (assuming these are common or specific to big_save structure)
VISUALIZE_FREQUENCY = 10 # Visualize validation examples every N epochs
CHECKPOINT_FREQUENCY = 1000 # Save standard checkpoints every N epochs
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


            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input_image, layer_idx)

                # Ensure shapes match for loss calculation
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

                # --- Calculate Loss Components ---
                if use_diff_as_target:
                    loss_mse = criterion_mse(output, target)
                else:
                    loss_mse = criterion_mse(input_image, output, target)

                if use_diff_as_target:
                    output_image = output + input_image
                    loss_perceptual = criterion_perceptual(output_image, original_target)
                else:
                    loss_perceptual = criterion_perceptual(output, original_target)


                loss = (lambda_mse * loss_mse) + (lambda_perceptual * loss_perceptual)

            total_loss += loss.item()
            total_mse_loss += loss_mse.item()
            total_perc_loss += loss_perceptual.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_mse = total_mse_loss / num_batches
    avg_perc = total_perc_loss / num_batches
    return avg_loss, avg_mse, avg_perc


# --- Training Function ---
def train_model(greyscale_param=True, subset_fraction_param=1.0): # Renamed params to avoid conflict
    """ Trains the LayerGeneratorHybrid model with train/val/test splits. """
    # Use global hyperparameters within this function
    global greyscale, subset_fraction, img_size, batch_size, num_epochs, learning_rate
    global cnn_start_filters, transformer_embed_dim, cnn_depth, transformer_layers, transformer_heads
    global weight_decay, lambda_mse, lambda_perceptual, use_diff_as_target
    global train_split, val_split, test_split, data_file, save_dir, best_model_filename
    global plot_filename, final_model_filename, run_timestamp, generate_image_set

    # Override from function parameters if needed, otherwise use global
    greyscale = greyscale_param
    subset_fraction = subset_fraction_param

    in_chans = 1 if greyscale else 4
    out_chans = in_chans

    print(f"--- Training Configuration ---")
    print(f"Timestamp: {run_timestamp}")
    print(f"Dataset: {data_file}, Greyscale: {greyscale}, Initial Subset: {subset_fraction*100:.1f}%")
    print(f"Splits: Train={train_split*100:.0f}%, Val={val_split*100:.0f}%, Test={test_split*100:.0f}%")
    print(f"Image Size: {img_size}x{img_size}, Batch Size: {batch_size}, Epochs: {num_epochs}")
    print(f"LR: {learning_rate}, Weight Decay: {weight_decay}")
    print(f"Use Difference as Target: {use_diff_as_target}")
    print(f"Generate Image Set (100 images post-training): {generate_image_set}")
    print(f"Saving results to: {save_dir}")
    print("-" * 30)

    os.makedirs(save_dir, exist_ok=True)
    plot_save_path = os.path.join(save_dir, plot_filename)
    best_model_path = os.path.join(save_dir, best_model_filename)
    final_model_path = os.path.join(save_dir, final_model_filename)

    print(f"Loading full dataset from: {data_file}")
    full_dataset = LayerPtDataset(data_file, img_size=img_size, transform=None,
                                  greyscale=greyscale, subset_fraction=subset_fraction)
    print(f"Successfully loaded dataset. Number of transitions: {len(full_dataset)}")

    if len(full_dataset) == 0:
        print("Error: Dataset contains no transitions. Exiting.")
        return

    if hasattr(full_dataset, 'num_layers') and full_dataset.num_layers is not None:
         num_layers_max = full_dataset.num_layers
         print(f"Using num_layers_max = {num_layers_max} (from dataset's {full_dataset.num_layers} layers) for embedding and rollouts.")
    else:
         # Attempt to infer from raw data if LayerPtDataset doesn't provide it directly
         try:
             temp_raw_data = torch.load(data_file, map_location='cpu')
             num_layers_max = temp_raw_data.shape[1] # N L C H W
             print(f"Inferred num_layers_max = {num_layers_max} from raw data file for embedding and rollouts.")
             del temp_raw_data
         except Exception as e:
             num_layers_max = 18 # Fallback default
             print(f"Warning: Could not reliably determine num_layers_max from dataset or raw file ({e}). Using default: {num_layers_max}")


    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    print(f"Splitting dataset: Train={train_size}, Validation={val_size}, Test={test_size}")
    if train_size == 0 or val_size == 0:
         print("Error: Train or Validation set size is 0. Cannot proceed.")
         return

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1) # ensure os.cpu_count is not None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)

    print(f"DataLoaders created: Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}")
    if len(train_loader) == 0:
        print("Error: Training dataloader has zero batches. Cannot train.")
        return

    model = LayerGeneratorHybrid(
        img_size=img_size, in_chans=in_chans, out_chans=out_chans,
        cnn_start_filters=cnn_start_filters, cnn_depth=cnn_depth,
        transformer_embed_dim=transformer_embed_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        num_layers_max=num_layers_max
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params / 1e6:.2f} M")

    if use_diff_as_target:
        criterion_mse = nn.MSELoss()
    else:
        criterion_mse = WeightedProportionalMSELoss(threshold=0.05).to(device)
    criterion_perceptual = VGGPerceptualLoss(feature_layers=[2, 7, 16, 25, 34]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"Using Automatic Mixed Precision (AMP): {use_amp}")

    print(f"\n--- Starting Training ---")
    start_train_time = time.time()

    train_losses_epoch = []
    val_losses_epoch = []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_total_train = 0.0
        epoch_loss_mse_train = 0.0
        epoch_loss_perc_train = 0.0
        num_train_batches = len(train_loader)
        progress_bar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)

        for batch_idx, batch_data in enumerate(progress_bar_train):
            input_image, target, layer_idx = batch_data
            original_target = target
            if use_diff_as_target:
                target = target - input_image

            input_image = input_image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            layer_idx = layer_idx.to(device, non_blocking=True)
            original_target = original_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input_image, layer_idx)
                if output.shape != target.shape:
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

                if use_diff_as_target:
                    loss_mse_train_batch = criterion_mse(output, target)
                else:
                    loss_mse_train_batch = criterion_mse(input_image, output, target)

                if use_diff_as_target:
                    output_image_train = output + input_image
                    loss_perceptual_train_batch = criterion_perceptual(output_image_train, original_target)
                else:
                    loss_perceptual_train_batch = criterion_perceptual(output, original_target)
                
                loss = (lambda_mse * loss_mse_train_batch) + (lambda_perceptual * loss_perceptual_train_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWarning: NaN/Inf train loss detected at epoch {epoch+1}, batch {batch_idx}. Skipping step.")
                optimizer.zero_grad(set_to_none=True) # Important to clear gradients that might be NaN/Inf
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss_total_train += loss.item()
            epoch_loss_mse_train += loss_mse_train_batch.item()
            epoch_loss_perc_train += loss_perceptual_train_batch.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar_train.set_postfix(
                loss=f"{loss.item():.4f}", mse=f"{loss_mse_train_batch.item():.4f}", perc=f"{loss_perceptual_train_batch.item():.4f}", lr=f"{current_lr:.1e}"
            )
        progress_bar_train.close()
        avg_train_loss = epoch_loss_total_train / num_train_batches if num_train_batches > 0 else 0.0
        avg_train_mse_comp = epoch_loss_mse_train / num_train_batches if num_train_batches > 0 else 0.0
        avg_train_perc_comp = epoch_loss_perc_train / num_train_batches if num_train_batches > 0 else 0.0
        train_losses_epoch.append(avg_train_loss)

        avg_val_loss, avg_val_mse_comp, avg_val_perc_comp = evaluate(
            model, val_loader, criterion_mse, criterion_perceptual, device, use_amp,
            lambda_mse, lambda_perceptual, use_diff_as_target, desc="Validating"
        )
        val_losses_epoch.append(avg_val_loss)
        
        # Weighted components for printing
        avg_train_mse_weighted = lambda_mse * avg_train_mse_comp
        avg_train_perc_weighted = lambda_perceptual * avg_train_perc_comp
        avg_val_mse_weighted = lambda_mse * avg_val_mse_comp
        avg_val_perc_weighted = lambda_perceptual * avg_val_perc_comp

        print(f"Epoch {epoch+1}/{num_epochs} -> "
              f"Train Loss: {avg_train_loss:.4f} (MSE: {avg_train_mse_weighted:.4f}, Perc: {avg_train_perc_weighted:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} (MSE: {avg_val_mse_weighted:.4f}, Perc: {avg_val_perc_weighted:.4f}) | "
              f"LR: {current_lr:.2e}")

        if SAVE_BEST_MODEL and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if use_amp else None,
                'best_val_loss': best_val_loss,
                'hyperparameters': {
                     'img_size': img_size, 'greyscale': greyscale, 'in_chans': in_chans, 'out_chans': out_chans,
                     'batch_size': batch_size, 'learning_rate': learning_rate,
                     'cnn_start_filters': cnn_start_filters, 'cnn_depth': cnn_depth,
                     'transformer_embed_dim': transformer_embed_dim, 'transformer_layers': transformer_layers,
                     'transformer_heads': transformer_heads, 'num_layers_max': num_layers_max,
                     'lambda_mse': lambda_mse, 'lambda_perceptual': lambda_perceptual,
                     'weight_decay': weight_decay, 'use_diff_as_target': use_diff_as_target,
                     'subset_fraction': subset_fraction # Save subset fraction as well
                }
            }
            torch.save(save_data, best_model_path)
            print(f"✅ New best model saved with Val Loss: {best_val_loss:.4f} at {best_model_path}")

        if (epoch + 1) % VISUALIZE_FREQUENCY == 0 or epoch == num_epochs - 1:
            print(f"--- Running Visualization for Epoch {epoch+1} ---")
            vis_batch = next(iter(val_loader), None)
            if vis_batch:
                vis_input, vis_target, vis_layer_idx = vis_batch
                vis_input = vis_input.to(device)
                vis_target = vis_target.to(device)
                vis_layer_idx = vis_layer_idx.to(device)
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                    vis_output_model = model(vis_input, vis_layer_idx)
                
                if use_diff_as_target:
                    target_to_show = vis_target - vis_input
                    output_to_show = vis_output_model
                else:
                    target_to_show = vis_target
                    output_to_show = vis_output_model
                
                save_image_grid(vis_input, target_to_show, output_to_show, epoch + 1, save_dir=save_dir,
                                use_diff_as_target=use_diff_as_target)
                print(f"Visualization grid saved for epoch {epoch+1}")
                model.train()
            else:
                print("Skipping visualization: Could not get validation batch.")

            if visualize_rollouts:
                 save_rollout_grid(model, save_dir, img_size, in_chans, device, epoch + 1,
                                   full_dataset,
                                   n_rollouts=min(5, batch_size),
                                   num_layers=num_layers_max,
                                   use_diff_as_target=use_diff_as_target)
                 print(f"Rollout visualization saved for epoch {epoch+1}")

        if SAVE_MODEL and (epoch + 1) % CHECKPOINT_FREQUENCY == 0:
             checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:03d}.pt')
             chkpt_data = {
                 'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'val_loss': avg_val_loss,
                 'hyperparameters': save_data['hyperparameters'] if 'save_data' in locals() and save_data else None
             }
             torch.save(chkpt_data, checkpoint_path)
             print(f"Checkpoint saved: {checkpoint_path}")

    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print("-" * 50)
    print(f"Training completed in {total_training_time / 60:.2f} minutes ({total_training_time:.1f} seconds).")

    test_loss_val = None # Initialize to keep track if test loss was computed
    if os.path.exists(best_model_path):
        print("\n--- Evaluating Best Model on Test Set ---")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Ensure all necessary hyperparameters are loaded or use current script's if not in checkpoint
        chkpt_hyperparams = checkpoint.get('hyperparameters', {})
        loaded_img_size = chkpt_hyperparams.get('img_size', img_size)
        loaded_in_chans = chkpt_hyperparams.get('in_chans', in_chans) # Ensure in_chans matches saved model
        loaded_out_chans = chkpt_hyperparams.get('out_chans', out_chans) # Ensure out_chans matches saved model
        loaded_cnn_start_filters = chkpt_hyperparams.get('cnn_start_filters', cnn_start_filters)
        loaded_cnn_depth = chkpt_hyperparams.get('cnn_depth', cnn_depth)
        loaded_transformer_embed_dim = chkpt_hyperparams.get('transformer_embed_dim', transformer_embed_dim)
        loaded_transformer_layers = chkpt_hyperparams.get('transformer_layers', transformer_layers)
        loaded_transformer_heads = chkpt_hyperparams.get('transformer_heads', transformer_heads)
        # Crucially, num_layers_max must match the model structure it was saved with.
        loaded_num_layers_max = chkpt_hyperparams.get('num_layers_max', num_layers_max)


        best_model_instance = LayerGeneratorHybrid(
            img_size=loaded_img_size,
            in_chans=loaded_in_chans,
            out_chans=loaded_out_chans,
            cnn_start_filters=loaded_cnn_start_filters,
            cnn_depth=loaded_cnn_depth,
            transformer_embed_dim=loaded_transformer_embed_dim,
            transformer_layers=loaded_transformer_layers,
            transformer_heads=loaded_transformer_heads,
            num_layers_max=loaded_num_layers_max
        )
        best_model_instance.load_state_dict(checkpoint['model_state_dict'])
        best_model_instance.to(device)
        best_model_instance.eval() # Explicitly set to eval mode

        # Use use_diff_as_target from checkpoint if available, otherwise from current script
        # This is important for correct evaluation logic.
        eval_use_diff_as_target = chkpt_hyperparams.get('use_diff_as_target', use_diff_as_target)


        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'N/A')} with Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
        print(f"Model config used for loading: img_size={loaded_img_size}, in_chans={loaded_in_chans}, num_layers_max={loaded_num_layers_max}, use_diff_as_target (for eval): {eval_use_diff_as_target}")


        test_loss_val, test_mse_comp, test_perc_comp = evaluate(
            best_model_instance, test_loader, criterion_mse, criterion_perceptual, device, use_amp,
            lambda_mse, lambda_perceptual, eval_use_diff_as_target, desc="Testing" # Use eval_use_diff_as_target
        )
        test_mse_weighted = lambda_mse * test_mse_comp
        test_perc_weighted = lambda_perceptual * test_perc_comp

        print("-" * 30)
        print(f"Final Test Set Performance (using best model):")
        print(f"  Test Loss: {test_loss_val:.4f}")
        print(f"  Test MSE (weighted): {test_mse_weighted:.4f} (Component: {test_mse_comp:.4f})")
        print(f"  Test Perceptual (weighted): {test_perc_weighted:.4f} (Component: {test_perc_comp:.4f})")
        print("-" * 30)

        # --- Generate Image Set using Best Model ---
        if generate_image_set:
            print("\n--- Generating Image Set (100 images) using Best Model ---")
            num_images_to_generate = 100
            gen_output_dir = os.path.join(save_dir, generated_images_folder_name)
            os.makedirs(gen_output_dir, exist_ok=True)
            

            # Load raw data for initial layer 0 images (as per generate_images.py)
            print(f"Loading raw data from: {data_file} for image generation start points.")
            try:
                raw_data_tensor = torch.load(data_file, map_location='cpu') # N L C H W
                data_num_samples_gen = raw_data_tensor.shape[0]
                # num_layers_for_rollout = raw_data_tensor.shape[1] # This should match loaded_num_layers_max

                # Ensure num_layers_for_rollout matches model's expectation (loaded_num_layers_max)
                # This is important as the model was trained with a specific num_layers_max for its embedding table.
                num_layers_for_rollout = loaded_num_layers_max
                print(f"Image generation will perform rollouts for {num_layers_for_rollout-1} steps (up to layer {num_layers_for_rollout-1}).")


                if not torch.is_floating_point(raw_data_tensor):
                    raw_data_tensor = raw_data_tensor.float() / 255.0 if raw_data_tensor.max() > 1.0 else raw_data_tensor.float()
                
                layer_0_data_gen = raw_data_tensor[:, 0, :, :, :] # Shape [samples, C, H, W]
                del raw_data_tensor # Free memory

                if data_num_samples_gen < num_images_to_generate:
                    print(f"Warning: Requested {num_images_to_generate} images, but only {data_num_samples_gen} unique starting images available. Using with replacement or fewer.")
                    start_indices_gen = random.choices(range(data_num_samples_gen), k=num_images_to_generate)
                else:
                    start_indices_gen = random.sample(range(data_num_samples_gen), num_images_to_generate)
                
                print(f"Selected {len(start_indices_gen)} random indices for starting images.")
                gen_start_time = time.time()

                for r_idx, sample_idx in enumerate(tqdm(start_indices_gen, desc="Generating Images")):
                    start_image_chw = layer_0_data_gen[sample_idx].cpu() # [C, H, W]
                    
                    if start_image_chw.shape[-1] != loaded_img_size:        # loaded_img_size == 64
                        start_image_chw = F.interpolate(
                            start_image_chw.unsqueeze(0),                   # add batch dim
                            size=(loaded_img_size, loaded_img_size),        # (64, 64)
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  
                    
                    current_layer_img = start_image_chw.unsqueeze(0).to(device) # [1, C, H, W]

                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                        for layer_idx_val in range(num_layers_for_rollout - 1): # Predict layer t+1 from layer t
                            input_image_gen = current_layer_img
                            layer_idx_tensor_gen = torch.tensor([layer_idx_val], dtype=torch.long, device=device)
                            
                            prediction = best_model_instance(input_image_gen, layer_idx_tensor_gen)

                            if eval_use_diff_as_target: # Use the same logic as during its training/evaluation
                                next_layer_img = input_image_gen + prediction
                            else:
                                next_layer_img = prediction
                            
                            next_layer_img = torch.clamp(next_layer_img, 0.0, 1.0)
                            current_layer_img = next_layer_img
                    
                    final_image_tensor = current_layer_img.squeeze(0).cpu()
                    save_path_gen = os.path.join(gen_output_dir, f"generated_final_img_{r_idx+1:03d}_from_sample{sample_idx}.png")
                    
                    # Handle channel swap for color images if needed (assuming model outputs BGR(A) if not greyscale)
                    # This matches the logic in the provided generate_images.py
                    if not greyscale and final_image_tensor.shape[0] >= 3: # Assuming greyscale comes from main script config
                        # If your model outputs RGB(A) directly, this swap might be unnecessary or need adjustment
                        # The original generate_images.py script implies BGR(A) -> RGB(A) for saving.
                        # If your dataset's LayerPtDataset already handles channel order for display/saving, adapt this.
                        # For now, assuming BGR(A) like format from model similar to user's script.
                        indices_swap = [2, 1, 0] + list(range(3, final_image_tensor.shape[0]))
                        try:
                            final_image_tensor_swapped = final_image_tensor[indices_swap, :, :]
                            save_image(final_image_tensor_swapped, save_path_gen)
                        except IndexError: # If fewer than 3 channels despite not greyscale (e.g. RG, RA), save as is.
                            print(f"Warning: Channel swap for color image failed for image {r_idx+1} (shape: {final_image_tensor.shape}). Saving without swap.")
                            save_image(final_image_tensor, save_path_gen)

                    elif greyscale:
                         save_image(final_image_tensor, save_path_gen)
                    else: # Fallback for unexpected channel counts
                         save_image(final_image_tensor, save_path_gen)


                gen_end_time = time.time()
                print(f"Finished generating {num_images_to_generate} images in {gen_end_time - gen_start_time:.2f} seconds.")
                print(f"Generated images saved in: {gen_output_dir}")

            except FileNotFoundError:
                print(f"Error: Data file {data_file} not found. Skipping generation of 100 images.")
            except Exception as e:
                print(f"An error occurred during image set generation: {e}")
                import traceback
                traceback.print_exc()


    else:
        print("Could not find the best model file to evaluate on the test set or generate images.")
        test_loss_val = None

    print(f"\nGenerating loss plot at {plot_save_path}")
    plot_start_epoch = 20 

    if num_epochs >= plot_start_epoch:
        epochs_to_plot = range(plot_start_epoch, num_epochs + 1)
        train_losses_to_plot = train_losses_epoch[plot_start_epoch - 1:]
        val_losses_to_plot = val_losses_epoch[plot_start_epoch - 1:]
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_to_plot, train_losses_to_plot, label=f'Training Loss (Epoch {plot_start_epoch}+)')
        plt.plot(epochs_to_plot, val_losses_to_plot, label=f'Validation Loss (Epoch {plot_start_epoch}+)')
        plt.title(f'Training and Validation Loss (Epochs {plot_start_epoch}-{num_epochs})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if test_loss_val is not None:
             plt.axhline(y=test_loss_val, color='r', linestyle='--', label=f'Final Test Loss ({test_loss_val:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        print(f"Loss plot saved (showing epochs {plot_start_epoch}-{num_epochs}).")
    else:
        print(f"Warning: Total epochs ({num_epochs}) is less than plot start epoch ({plot_start_epoch}). Plotting all epochs.")
        epochs_all = range(1, num_epochs + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_all, train_losses_epoch, label='Training Loss')
        plt.plot(epochs_all, val_losses_epoch, label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs (Full Range)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if test_loss_val is not None:
             plt.axhline(y=test_loss_val, color='r', linestyle='--', label=f'Final Test Loss ({test_loss_val:.4f})')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_save_path)
        print("Loss plot saved (full range shown due to short training).")

    if SAVE_MODEL:
        # Save the final model state (potentially different from best_model if training continued)
        final_model_save_data = {
            'epoch': num_epochs, # Current epoch is num_epochs
            'model_state_dict': model.state_dict(), # Current model state
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'final_val_loss': val_losses_epoch[-1] if val_losses_epoch else None,
             'hyperparameters': { # Store key hyperparameters
                     'img_size': img_size, 'greyscale': greyscale, 'in_chans': in_chans, 'out_chans': out_chans,
                     'batch_size': batch_size, 'learning_rate': learning_rate,
                     'cnn_start_filters': cnn_start_filters, 'cnn_depth': cnn_depth,
                     'transformer_embed_dim': transformer_embed_dim, 'transformer_layers': transformer_layers,
                     'transformer_heads': transformer_heads, 'num_layers_max': num_layers_max,
                     'lambda_mse': lambda_mse, 'lambda_perceptual': lambda_perceptual,
                     'weight_decay': weight_decay, 'use_diff_as_target': use_diff_as_target,
                     'subset_fraction': subset_fraction
                }
        }
        torch.save(final_model_save_data, final_model_path)
        print(f"Final model state (with metadata) saved as {final_model_path}")

    print(f"Results (best model, plot, checkpoints, generated images if any) saved in: {save_dir}")
    print("-" * 50)


if __name__ == '__main__':
    # Use global hyperparameters directly, or pass them if you prefer stricter scoping.
    # For this modification, train_model now accesses global hyperparameters.
    train_model(greyscale_param=greyscale, subset_fraction_param=subset_fraction)
    print("Training script finished.")