import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from unet import UNet


def get_dataloaders(batch_size=128):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def add_noise(images, noise_factor=0.4):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


def train_one_epoch(model, loader, optimizer, criterion, device, denoising=False, noise_factor=0.4):
    model.train()
    total_loss = 0.0

    for images, _ in loader:
        images = images.to(device)
        inputs = add_noise(images, noise_factor) if denoising else images

        outputs = model(inputs)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, denoising=False, noise_factor=0.4):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            inputs = add_noise(images, noise_factor) if denoising else images

            outputs = model(inputs)
            loss = criterion(outputs, images)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def save_results(model, loader, device, filename, denoising=False, noise_factor=0.4, n=8):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:n].to(device)

    inputs = add_noise(images, noise_factor) if denoising else images

    with torch.no_grad():
        reconstructed = model(inputs)

    images = images.cpu()
    inputs = inputs.cpu()
    reconstructed = reconstructed.cpu()

    rows = 3 if denoising else 2
    fig, axes = plt.subplots(rows, n, figsize=(n * 1.5, rows * 1.5))

    for i in range(n):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', loc='left')

        if denoising:
            axes[1, i].imshow(inputs[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Noisy', loc='left')

            axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Denoised', loc='left')
        else:
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', loc='left')

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved visual sample to {filename}")


def run_experiment(name, denoising=False, epochs=5, lr=1e-3, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Running {name} on {device} ---")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, denoising=denoising)
        test_loss = evaluate(model, test_loader, criterion, device, denoising=denoising)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    save_path = f"{name.lower().replace(' ', '_')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    plot_path = f"{name.lower().replace(' ', '_')}_results.png"
    save_results(model, test_loader, device, plot_path, denoising=denoising)

    return model


if __name__ == '__main__':
    run_experiment(name="Standard UNet AE", denoising=False, epochs=5)
    run_experiment(name="Denoising UNet AE", denoising=True, epochs=5)
