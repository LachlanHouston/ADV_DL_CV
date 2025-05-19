import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import random

# Ensure project structure allows imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.models.model_unet_transbottle import LayerGeneratorHybrid

MODEL = "small_perc"   # "big" or "big_diff" or "small_perc"

if MODEL == "big_diff":
    # Configuration for evaluation (must match training)
    model_path = 'results/8Big_diff_20250425_221116/hybrid_layer_generator_final.pth'
    data_file = 'data/full_dataset_128.pt'
    greyscale = False
    img_size = 128
    cnn_start_filters = 64
    cnn_depth = 4
    transformer_embed_dim = 512
    transformer_layers = 8
    transformer_heads = 8
    use_diff_as_target = True

    output_dir = 'generations_big_diff/' # Output directory for final images

elif MODEL == "big":
    # Configuration for evaluation (must match training)
    model_path = 'results/5big_20250422_121007/hybrid_layer_generator_final.pth'
    model_path = 'results/5big_20250422_121007/checkpoint_epoch_200.pt'
    data_file = 'data/full_dataset_128.pt'
    greyscale = False
    img_size = 128
    cnn_start_filters = 64
    cnn_depth = 4
    transformer_embed_dim = 512
    transformer_layers = 8
    transformer_heads = 8
    use_diff_as_target = False

    output_dir = 'generations_big/' # Output directory for final images

elif MODEL == "small_perc":
    model_path = "results/20250509_204640_small_perc/best_model_val.pth"
    data_file = 'data/full_dataset_color.pt'
    greyscale = False         
    img_size = 64             
    cnn_start_filters = 32    
    cnn_depth = 4             
    transformer_embed_dim = 256 
    transformer_layers = 6    
    transformer_heads = 8     
    use_diff_as_target = False 
    
    output_dir = 'generations_small_perc/'


# Evaluation specific settings
num_rollouts = 100

# rollout_prefix = "layer0_rollout" # No longer needed for subdirectory

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load raw data tensor (assuming N L C H W format)
print(f"Loading raw data tensor from: {data_file}")
raw_data = torch.load(data_file, map_location='cpu')
print(f"Successfully loaded data tensor.")

# Unpack dimensions according to N L C H W
data_num_samples, data_num_layers, data_C, data_H, data_W = raw_data.shape
print(f"Data shape: [Samples={data_num_samples}, Layers={data_num_layers}, C={data_C}, H={data_H}, W={data_W}]")

# Derive num_layers_max for rollout loop control
num_layers_max = data_num_layers
print(f"Using num_layers_max = {num_layers_max} from data file for rollout length.")

# Ensure data is float
if not torch.is_floating_point(raw_data):
    print("Converting data tensor to float.")
    raw_data = raw_data.float()
    # Add normalization here if needed, e.g. / 255.0

# Load model state
print(f"Loading model state from: {model_path}")
# Calculate in_chans/out_chans based on configuration
in_chans = 1 if greyscale else 4
out_chans = in_chans
print(f"Instantiating model with configured params: img_size={img_size}, in_chans={in_chans}, out_chans={out_chans}")

# Instantiate model using parameters from configuration
model = LayerGeneratorHybrid(
    img_size=img_size,
    in_chans=in_chans,
    out_chans=out_chans,
    cnn_start_filters=cnn_start_filters,
    cnn_depth=cnn_depth,
    transformer_embed_dim=transformer_embed_dim,
    transformer_layers=transformer_layers,
    transformer_heads=transformer_heads,
    num_layers_max=num_layers_max
)

# Load the state dictionary
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval() # Set model to evaluation mode
print("Model loaded successfully.")


# Prepare for rollouts
# Extract Layer 0 data: Shape [samples, C, H, W]
layer_0_data = raw_data[:, 0, :, :, :]
# Free memory
del raw_data

# Select random indices for starting images
start_indices = random.sample(range(data_num_samples), num_rollouts)
print(f"Selected {num_rollouts} random indices from Layer 0.")


# Generate rollouts
print(f"\n--- Starting Generation of {num_rollouts} Rollouts from Layer 0 ---")
rollout_start_time = time.time()

for r_idx in range(num_rollouts):
    # Get index for starting image
    sample_idx = start_indices[r_idx]
    print(f"\nGenerating Rollout {r_idx + 1}/{num_rollouts} (from Sample Index {sample_idx})...")
    # Removed subdirectory creation

    # 1. Get the initial Layer 0 image
    # Shape is [C, H, W]
    start_image_chw = layer_0_data[sample_idx].cpu()

    # Assume dimensions match config, add batch dim [1, C, H, W], move to device
    current_layer_img = start_image_chw.unsqueeze(0).to(device)

    # Removed saving of initial layer

    # 2. Perform the rollout step-by-step
    with torch.no_grad(): # Disable gradients during inference
        progress_bar = tqdm(range(num_layers_max - 1), desc=f"Rollout {r_idx+1}", leave=False)
        for layer_idx_val in progress_bar: # Predict layer t+1 from layer t
            # Prepare input image and layer index tensor
            input_image = current_layer_img
            layer_idx_tensor = torch.tensor([layer_idx_val], dtype=torch.long, device=device)

            # Predict next step
            prediction = model(input_image, layer_idx_tensor)

            # Calculate image for the next layer (t+1)
            if use_diff_as_target:
                next_layer_img = input_image + prediction
            else:
                next_layer_img = prediction

            # Clamp output to valid range
            next_layer_img = torch.clamp(next_layer_img, 0.0, 1.0)

            # Removed saving of intermediate layers

            # Update current image for next iteration
            current_layer_img = next_layer_img

    progress_bar.close()

    # 3. Save ONLY the final layer image after the loop
    final_image_tensor = current_layer_img.squeeze(0).cpu() # Get final image, remove batch, move to CPU

    # Define save path directly in the output directory
    save_path = os.path.join(output_dir, f"final_layer_{r_idx+1:03d}_idx{sample_idx}.png")

    # Swap channels if BGR(A) -> RGB(A) for saving color images
    if not greyscale and final_image_tensor.shape[0] >= 3:
        indices = [2, 1, 0] + list(range(3, final_image_tensor.shape[0]))
        final_image_tensor = final_image_tensor[indices, :, :]

    # Save the final image
    save_image(final_image_tensor, save_path)
    print(f"Rollout {r_idx + 1} finished. Final image saved: {save_path}")

# End of rollouts
rollout_end_time = time.time()
total_rollout_time = rollout_end_time - rollout_start_time
print("-" * 50)
print(f"Generated {num_rollouts} final images in {total_rollout_time / 60:.2f} minutes ({total_rollout_time:.1f} seconds).")
print(f"All final images saved within the '{output_dir}' directory.")
print("-" * 50)

# Main execution block
if __name__ == '__main__':
    pass