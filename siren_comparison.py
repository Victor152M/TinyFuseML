import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(30 * x)

# SIREN Layer with init from Sitzmann et al.
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                            1 / self.linear.in_features)
            else:
                bound = np.sqrt(6 / self.linear.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# Full SIREN model
class SIREN(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3):
        super().__init__()
        layers = [SineLayer(in_features, hidden_features, is_first=True)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))
        layers.append(nn.Sigmoid())  # Output RGB in [0,1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def load_image(path):
    img = Image.open(path).convert('RGB')
    transform = T.ToTensor()
    return transform(img)

def create_coords(H, W):
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    grid = torch.stack(torch.meshgrid(y, x, indexing='ij'), -1)  # [H,W,2]
    return grid.reshape(-1, 2)

def save_image(tensor, path):
    img = (tensor * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def train_siren(image_path, batch_size=65536, target_loss=0.0005, max_iters=2000, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = load_image(image_path)  # [3,H,W]
    C, H, W = img.shape
    coords = create_coords(H, W).to(device)
    pixels = img.permute(1, 2, 0).reshape(-1, 3).to(device)

    model = SIREN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    num_pixels = coords.shape[0]

    print(f"Training on image: {H}x{W} pixels")

    for iteration in range(max_iters):
        batch_idx = torch.randperm(num_pixels, device=device)[:batch_size]
        batch_coords = coords[batch_idx]
        batch_pixels = pixels[batch_idx]

        optimizer.zero_grad()
        preds = model(batch_coords)
        loss = loss_fn(preds, batch_pixels)
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iter {iteration}: Loss = {loss.item():.6f}")

        if loss.item() < target_loss:
            print(f"Target loss {target_loss} reached at iteration {iteration}")
            break

    # Reconstruct image
    with torch.no_grad():
        preds_full = []
        chunk_size = 100000
        for i in range(0, num_pixels, chunk_size):
            preds_chunk = model(coords[i:i+chunk_size])
            preds_full.append(preds_chunk)
        preds_full = torch.cat(preds_full, dim=0)

    img_pred = preds_full.reshape(H, W, 3).cpu().numpy()
    save_image(img_pred, "reconstructed_image.png")
    print("Saved reconstructed_image.png")

if __name__ == "__main__":
    train_siren("training_data/mystery_shack.png")
