import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import time
import encoding
import time

class MLP(nn.Module):
    def __init__(self, in_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = T.ToTensor()(img)
    C, H, W = img.shape
    pixels = img.permute(1, 2, 0).reshape(-1, 3)
    return pixels, H, W

def coords(H, W):
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


@torch.no_grad()
def reconstruct(model, enc, xy, H, W):
    pred = model(enc(xy))
    img = pred.reshape(H, W, 3).cpu().numpy()
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def train(img_path, iters=300, batch_size=2**21, lr=0.0002):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pixels, H, W = load_image(img_path)
    pixels = pixels.to(device)

    xy = coords(H, W).to(device)

    enc = encoding.MultiResHashGrid(2).to(device)
    model = MLP(enc.output_dim).to(device)

    opt = torch.optim.SGD(
        list(model.parameters()) + list(enc.parameters()),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    N = xy.shape[0]
    total_pixels = 0
    start_time = time.time()

    for i in range(iters):

        idx = torch.randint(0, N, (batch_size,), device=device)

        xb = xy[idx]
        yb = pixels[idx]

        feat = enc(xb)
        pred = model(feat)

        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_pixels += xb.shape[0]
        
        if i % 50 == 0:
            print(i, loss.item())


    torch.cuda.synchronize() if device == "cuda" else None

    end_time = time.time()
    elapsed = end_time - start_time

    pixels_per_sec = total_pixels / elapsed
    mega_pixels_per_sec = pixels_per_sec / 1e6

    #print("\n--- THROUGHPUT ---")
    #print(f"Time: {elapsed:.4f} sec")
    #print(f"Pixels/sec: {pixels_per_sec:.2f}")
    #print(f"M Pixels/sec: {mega_pixels_per_sec:.3f}")

    print("Reconstructing...")

    model.eval()
    enc.eval()

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        pred = model(enc(xy))

    torch.cuda.synchronize()
    end_time = time.time()

    elapsed = end_time - start_time

    pixels_per_sec = (H * W) / elapsed
    mega_pixels_per_sec = pixels_per_sec / 1e6

    print("\n--- THROUGHPUT ---")
    print(f"Time: {elapsed:.4f} sec")
    print(f"Pixels/sec: {pixels_per_sec:.2f}")
    print(f"M Pixels/sec: {mega_pixels_per_sec:.3f}")

    img = pred.reshape(H, W, 3).cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    Image.fromarray(img).save("result.png")
    print("Saved: result.png")

if __name__ == "__main__":
    train("images/cassette_shop_fullhd.png")