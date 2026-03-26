from PIL import Image
import numpy as np
from skimage.metrics import mean_squared_error
import math

def psnr_chunked(original_path, reconstructed_path, chunk_size=8192):
    Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check

    orig_img = Image.open(original_path)
    recon_img = Image.open(reconstructed_path)

    width, height = orig_img.size
    n_pixels = 0
    mse_sum = 0.0

    # Loop over tiles
    for top in range(0, height, chunk_size):
        for left in range(0, width, chunk_size):
            bottom = min(top + chunk_size, height)
            right = min(left + chunk_size, width)

            orig_tile = np.array(orig_img.crop((left, top, right, bottom)), dtype=np.float32)
            recon_tile = np.array(recon_img.crop((left, top, right, bottom)), dtype=np.float32)

            diff = orig_tile - recon_tile
            mse_sum += np.sum(diff ** 2)
            n_pixels += orig_tile.size  # total number of values (pixels × channels)

    mse = mse_sum / n_pixels
    psnr_value = 10 * np.log10((255.0 ** 2) / mse)
    return psnr_value



ORIGINAL_PATH = "training_data/girl_with_pearl.png"
RECONSTRUCTION_PATH = "results/girl_with_pearl_full_inference.png"

value = psnr_chunked(ORIGINAL_PATH, RECONSTRUCTION_PATH)
print("PSNR:", value)