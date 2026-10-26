import os
import glob
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

from unet import UNet


class CelebADataset(Dataset):
    def __init__(self, root_dir='./data/celeba', transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_dataloaders(root_dir='./data/celeba', batch_size=32, train_ratio=0.85):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = CelebADataset(root_dir=root_dir, transform=transform)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images in loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_results(model, loader, device, filename='celeba_unet_ae_results.png', n=8):
    model.eval()
    images = next(iter(loader))[:n].to(device)

    with torch.no_grad():
        reconstructed = model(images)

    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))

    for i in range(n):
        # Original
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', loc='left', fontsize=12)

        # Reconstructed
        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', loc='left', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved visual sample to {filename}")


def run_training(epochs=15, lr=1e-3, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training UNet Autoencoder on Celebrity Dataset using {device} ---")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    save_path = 'celeba_unet_ae.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    save_results(model, val_loader, device, 'celeba_unet_ae_results.png')
    return model


if __name__ == '__main__':
    run_training(epochs=15)
