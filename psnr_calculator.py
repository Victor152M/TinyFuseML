from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


original = np.array(Image.open("images/cassette_shop_fullhd.png"))
reconstructed = np.array(Image.open("build/Full_Inference.png"))

value = psnr(original, reconstructed)
print("PSNR:", value)
