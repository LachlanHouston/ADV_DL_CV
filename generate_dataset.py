import cv2
import numpy as np
import h5py
from generate_character import get_character, get_random_character_options
from tqdm import tqdm

# Global configuration variables.
DATASET_SIZE = 1000          # Total number of characters to generate.
OUTPUT_HDF5 = f'kenney_dataset_{DATASET_SIZE}.h5'  # HDF5 file to store the dataset.
BASE_PATH = 'kenney_modular-characters/Spritesheet/'

def main():
    # Create an HDF5 file to hold the dataset.
    # Each sample has 18 progressive images of size 600x600 with 4 channels (RGBA).
    with h5py.File(OUTPUT_HDF5, 'w') as hf:
        dataset = hf.create_dataset(
            "images",
            shape=(DATASET_SIZE, 18, 600, 600, 4),
            dtype=np.uint8,
            compression="gzip"
        )
        
        for i in tqdm(range(DATASET_SIZE), desc="Generating dataset", unit="sample"):
            # Randomly generate character options.
            options = get_random_character_options()
            # Generate all 18 progressive layers.
            layers = get_character(options, layer_progression=True, num_layers=18, verbose=False, base_path=BASE_PATH)
            
            # Store each progressive layer into the HDF5 dataset.
            for j, layer in enumerate(layers):
                dataset[i, j, :, :, :] = layer
        
        print("Dataset generation complete. Saved to", OUTPUT_HDF5)

if __name__ == "__main__":
    main()