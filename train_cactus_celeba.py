import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from cactus import CactusImageAutoencoder
from train_celeba import get_dataloaders


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


def save_results(model, loader, device, filename='cactus_celeba_results.png', n=8):
    model.eval()
    images = next(iter(loader))[:n].to(device)

    with torch.no_grad():
        reconstructed = model(images)

    images = images.cpu().permute(0, 2, 3, 1).numpy()
    reconstructed = reconstructed.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))

    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', loc='left', fontsize=12)

        axes[1, i].imshow(reconstructed[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Cactus Recon', loc='left', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved visual sample to {filename}")


def run_training(epochs=20, lr=1e-3, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training Cactus Transformer Autoencoder on Celebrity Dataset ({device}) ---")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)
    model = CactusImageAutoencoder(
        img_size=64,
        patch_size=8,
        in_channels=3,
        d_model=128,
        num_heads=4,
        num_layers=3
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    save_path = 'cactus_celeba.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    save_results(model, val_loader, device, 'cactus_celeba_results.png')
    return model


if __name__ == '__main__':
    run_training(epochs=20)
