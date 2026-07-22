from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

# 1. Disable PIL decompression bomb limit
Image.MAX_IMAGE_PIXELS = None

# 2. Load images and convert to RGB (drops alpha if present)
orig = Image.open("images/cassette_shop_fullhd.png").convert("RGB")
recon = Image.open("build/Full_Inference.png").convert("RGB")

# 3. Convert to numpy arrays
original = np.array(orig)
reconstructed = np.array(recon)


# 5. They should now have identical shape if dimensions match.
#    If they still differ (e.g., one is downscaled), you can resize:
if original.shape != reconstructed.shape:
    from skimage.transform import resize
    reconstructed = resize(
        reconstructed,
        original.shape[:2],
        preserve_range=True,
        anti_aliasing=True
    ).astype(np.uint8)

# 6. Compute PSNR
value = psnr(original, reconstructed)
print("PSNR:", value)
