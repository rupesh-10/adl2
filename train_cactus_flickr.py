import os
import re
import json
import random
import argparse
import textwrap
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from cactus_multimodal import CactusMultimodal


class SimpleTokenizer:
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    unk_token = '<unk>'

    def __init__(self, captions, max_vocab=2500, max_len=28):
        words = []
        for c in captions:
            words.extend(re.findall(r'[a-z0-9]+', c.lower()))

        counts = Counter(words)
        vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token] + [
            w for w, _ in counts.most_common(max_vocab - 4)
        ]

        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for i, w in enumerate(vocab)}
        self.vocab_size = len(vocab)
        self.max_len = max_len

        self.pad_id = self.word2id[self.pad_token]
        self.sos_id = self.word2id[self.sos_token]
        self.eos_id = self.word2id[self.eos_token]
        self.unk_id = self.word2id[self.unk_token]

    def encode(self, text):
        words = re.findall(r'[a-z0-9]+', text.lower())
        tokens = [self.sos_id] + [self.word2id.get(w, self.unk_id) for w in words[:self.max_len - 2]] + [self.eos_id]
        if len(tokens) < self.max_len:
            tokens += [self.pad_id] * (self.max_len - len(tokens))
        return tokens

    def decode(self, token_ids):
        words = []
        pad_id = getattr(self, 'pad_id', 0)
        sos_id = getattr(self, 'sos_id', 1)
        eos_id = getattr(self, 'eos_id', 2)
        unk_id = getattr(self, 'unk_id', 3)
        special_ids = {pad_id, sos_id, unk_id}

        for idx in token_ids:
            if idx in special_ids:
                continue
            if idx == eos_id:
                break
            w = self.id2word.get(idx, '')
            if w and w not in ('<pad>', '<sos>', '<eos>', '<unk>'):
                words.append(w)
        return ' '.join(words)


class FlickrDataset(Dataset):
    def __init__(self, data_dir='./data/flickr30k', tokenizer=None, transform=None, pairs=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'images')
        if pairs is not None:
            self.pairs = pairs
        else:
            with open(os.path.join(data_dir, 'captions.json')) as f:
                self.pairs = json.load(f)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        img_path = os.path.join(self.img_dir, item['image'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        tokens = torch.tensor(self.tokenizer.encode(item['caption']), dtype=torch.long)
        return image, tokens, item['caption'], item['image']


def get_dataloaders(data_dir='./data/flickr30k', batch_size=16, max_samples=None, train_ratio=0.85):
    with open(os.path.join(data_dir, 'captions.json')) as f:
        all_pairs = json.load(f)

    if max_samples is not None and max_samples < len(all_pairs):
        all_pairs = all_pairs[:max_samples]

    captions = [p['caption'] for p in all_pairs]
    tokenizer = SimpleTokenizer(captions, max_vocab=3500, max_len=28)

    train_transform = transforms.Compose([
        transforms.Resize((68, 68)),
        transforms.RandomCrop((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    full_dataset_raw = FlickrDataset(data_dir=data_dir, tokenizer=tokenizer, pairs=all_pairs)
    total_len = len(full_dataset_raw)
    train_size = int(train_ratio * total_len)

    indices = list(range(total_len))
    rng = random.Random(42)
    rng.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    class SubDataset(Dataset):
        def __init__(self, base_dataset, sub_indices, transform):
            self.base = base_dataset
            self.indices = sub_indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            item = self.base.pairs[self.indices[idx]]
            img_path = os.path.join(self.base.img_dir, item['image'])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            tokens = torch.tensor(self.base.tokenizer.encode(item['caption']), dtype=torch.long)
            return image, tokens, item['caption'], item['image']

    train_data = SubDataset(full_dataset_raw, train_indices, train_transform)
    val_data = SubDataset(full_dataset_raw, val_indices, val_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer, val_data


def info_nce_loss(e_S, e_I, temperature=0.15):
    """Bidirectional InfoNCE contrastive alignment to prevent representation collapse."""
    z_S = F.normalize(e_S, dim=-1)
    z_I = F.normalize(e_I, dim=-1)
    sim = torch.matmul(z_S, z_I.T) / temperature  # (B, B)
    labels = torch.arange(e_S.size(0), device=e_S.device)
    loss_s2i = F.cross_entropy(sim, labels)
    loss_i2s = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_s2i + loss_i2s)


def calc_ssim(x, y, window_size=7):
    """Structural Similarity Index (SSIM) to directly enforce shape & contours."""
    channel = x.size(1)
    w = torch.ones(channel, 1, window_size, window_size, device=x.device) / (window_size * window_size)
    mu_x = F.conv2d(x, w, padding=window_size//2, groups=channel)
    mu_y = F.conv2d(y, w, padding=window_size//2, groups=channel)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, w, padding=window_size//2, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=window_size//2, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=window_size//2, groups=channel) - mu_xy
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()


def calc_laplacian_pyramid_loss(pred, target):
    """Isolates high-frequency spatial components to prevent blurry / flat synthesis."""
    kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]], device=pred.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    pred_blur = F.conv2d(pred, kernel, padding=1, groups=3)
    target_blur = F.conv2d(target, kernel, padding=1, groups=3)
    return F.l1_loss(pred - pred_blur, target - target_blur)


def calc_image_loss(pred, target):
    """Multi-Scale Structural & Frequency Loss: L1 + SSIM Shape + Laplacian Details + Edge Gradients."""
    l1 = F.l1_loss(pred, target)
    ssim_val = calc_ssim(pred, target)
    ssim_loss = 1.0 - ssim_val

    # Laplacian high-frequency edge & texture loss (sharpens shadows and object details)
    lap_loss = calc_laplacian_pyramid_loss(pred, target)

    # Spatial edge gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    grad = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    # Global color & contrast matching
    color_mean = F.l1_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))
    color_std = F.l1_loss(pred.std(dim=[2, 3]), target.std(dim=[2, 3]))

    return l1 + 0.8 * ssim_loss + 0.6 * lap_loss + 0.4 * grad + 0.3 * (color_mean + color_std)


def train_one_epoch(model, loader, optimizer, tokenizer, device):
    model.train()
    total_t2i_loss = 0.0
    total_i2i_loss = 0.0
    total_i2t_loss = 0.0
    total_align_loss = 0.0

    for images, tokens, _, _ in loader:
        images = images.to(device)
        tokens = tokens.to(device)

        # 1. Text to Image generation (with prefix cross-attention, AdaMod & ResBlock image head)
        pred_images = model.forward_t2i(tokens, pad_id=tokenizer.pad_id)
        loss_t2i = calc_image_loss(pred_images, images)

        # 2. Image to Image autoencoding (trains visual decoder & ResBlocks for detail)
        recon_images = model.forward_i2i(images)
        loss_i2i = calc_image_loss(recon_images, images)

        # 3. Image to Text caption prediction with token dropout regularization
        inp_tokens = tokens[:, :-1].clone()
        tgt_tokens = tokens[:, 1:]

        mask_prob = 0.10
        rand_mask = (torch.rand_like(inp_tokens.float()) < mask_prob) & (inp_tokens != tokenizer.pad_id) & (inp_tokens != tokenizer.sos_id)
        inp_tokens[rand_mask] = tokenizer.unk_id

        logits = model.forward_i2t(images, inp_tokens)
        loss_i2t = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            tgt_tokens.reshape(-1),
            ignore_index=tokenizer.pad_id,
            label_smoothing=0.04
        )

        # 4. Bidirectional InfoNCE Contrastive Alignment
        _, e_S = model.encode_text(tokens, pad_id=tokenizer.pad_id)
        _, e_I = model.encode_image(images)
        loss_align = info_nce_loss(e_S, e_I, temperature=0.15)

        loss = 2.0 * loss_t2i + 1.0 * loss_i2i + 0.6 * loss_i2t + 0.3 * loss_align

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_t2i_loss += loss_t2i.item() * images.size(0)
        total_i2i_loss += loss_i2i.item() * images.size(0)
        total_i2t_loss += loss_i2t.item() * images.size(0)
        total_align_loss += loss_align.item() * images.size(0)

    n = len(loader.dataset)
    return total_t2i_loss / n, total_i2i_loss / n, total_i2t_loss / n, total_align_loss / n


def evaluate(model, loader, tokenizer, device):
    model.eval()
    total_t2i_loss = 0.0
    total_i2i_loss = 0.0
    total_i2t_loss = 0.0
    total_align_loss = 0.0

    with torch.no_grad():
        for images, tokens, _, _ in loader:
            images = images.to(device)
            tokens = tokens.to(device)

            pred_images = model.forward_t2i(tokens, pad_id=tokenizer.pad_id)
            loss_t2i = calc_image_loss(pred_images, images)

            recon_images = model.forward_i2i(images)
            loss_i2i = calc_image_loss(recon_images, images)

            inp_tokens = tokens[:, :-1]
            tgt_tokens = tokens[:, 1:]
            logits = model.forward_i2t(images, inp_tokens)
            loss_i2t = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                tgt_tokens.reshape(-1),
                ignore_index=tokenizer.pad_id
            )

            _, e_S = model.encode_text(tokens, pad_id=tokenizer.pad_id)
            _, e_I = model.encode_image(images)
            loss_align = info_nce_loss(e_S, e_I, temperature=0.15)

            total_t2i_loss += loss_t2i.item() * images.size(0)
            total_i2i_loss += loss_i2i.item() * images.size(0)
            total_i2t_loss += loss_i2t.item() * images.size(0)
            total_align_loss += loss_align.item() * images.size(0)

    n = len(loader.dataset)
    return total_t2i_loss / n, total_i2i_loss / n, total_i2t_loss / n, total_align_loss / n


def save_image_to_text_results(model, val_dataset, tokenizer, device, filename='cactus_flickr_i2t_results.png', n=6, chosen_indices=None, seed=None):
    """
    Renders dedicated Image-to-Text (I2T) result figure:
    - 2x3 card grid of 6 test samples
    - Left: Input Image (with sample ID & filename)
    - Right: Ground Truth caption in blue card & Model Predicted caption in green card (no text clipping)
    """
    model.eval()
    if chosen_indices is None:
        if seed is not None:
            rng = random.Random(seed)
            chosen_indices = rng.sample(range(len(val_dataset)), min(n, len(val_dataset)))
        else:
            chosen_indices = [6, 7, 9, 10, 15, 54] if len(val_dataset) > 54 else list(range(min(n, len(val_dataset))))

    batch_imgs, batch_caps, batch_names = [], [], []
    for idx in chosen_indices:
        img, tok, cap, name = val_dataset[idx]
        batch_imgs.append(img)
        batch_caps.append(cap)
        batch_names.append(name)

    images = torch.stack(batch_imgs).to(device)

    with torch.no_grad():
        predicted_captions = model.generate_caption(images, tokenizer, max_len=28, repetition_penalty=1.4, beam_width=4, min_len=6)

    orig_np = images.cpu().permute(0, 2, 3, 1).numpy()

    fig = plt.figure(figsize=(18, 8.5), facecolor='#ffffff')
    fig.suptitle('Cactus Multimodal Transformer: Image-to-Text (I2T) Captioning Results (Flickr 30k)\n'
                 'Conditioning: Input Image -> Autoregressive Beam Search Prediction vs. Ground Truth Caption',
                 fontsize=13, fontweight='bold', color='#1a365d', y=0.97)

    outer = fig.add_gridspec(2, 3, wspace=0.15, hspace=0.22, left=0.03, right=0.97, top=0.88, bottom=0.04)

    for idx in range(min(n, len(chosen_indices))):
        r, c = idx // 3, idx % 3
        sub = outer[r, c].subgridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.08)

        # Left: Input Image
        ax_img = fig.add_subplot(sub[0])
        ax_img.imshow(orig_np[idx])
        ax_img.axis('off')
        ax_img.set_title(f'Sample {idx+1}: {batch_names[idx]}\n[INPUT IMAGE]', fontsize=8.8, fontweight='bold', color='#2d3748', pad=4)

        # Right: Captions Display Panel
        ax_txt = fig.add_subplot(sub[1])
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis('off')

        gt_clean = batch_caps[idx].strip('\"\'')
        pred_clean = predicted_captions[idx].strip('\"\'')
        gt_wrap = textwrap.fill(gt_clean, width=28)
        pred_wrap = textwrap.fill(pred_clean, width=28)

        ax_txt.text(0.02, 0.72, f'GROUND TRUTH:\n\"{gt_wrap}\"',
                    va='center', ha='left', fontsize=8.5, color='#1a202c',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ebf8ff', edgecolor='#90cdf4', lw=1.2))

        ax_txt.text(0.02, 0.28, f'MODEL PREDICTED (I2T):\n\"{pred_wrap}\"',
                    va='center', ha='left', fontsize=8.5, color='#1a202c',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0fff4', edgecolor='#9ae6b4', lw=1.2))

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Image-to-Text (I2T) result figure to {filename}")
    return predicted_captions, chosen_indices


def save_text_to_image_results(model, val_dataset, tokenizer, device, filename='cactus_flickr_t2i_results.png', n=6, chosen_indices=None, seed=None):
    """
    Renders dedicated Text-to-Image (T2I) result figure:
    - Row 0: Input Text Prompt Cards
    - Row 1: Target Real Images (Ground Truth)
    - Row 2: Cactus Synthesized Images (Generated from Prompt)
    """
    model.eval()
    if chosen_indices is None:
        if seed is not None:
            rng = random.Random(seed)
            chosen_indices = rng.sample(range(len(val_dataset)), min(n, len(val_dataset)))
        else:
            chosen_indices = [6, 7, 9, 10, 15, 54] if len(val_dataset) > 54 else list(range(min(n, len(val_dataset))))

    batch_imgs, batch_tokens, batch_caps, batch_names = [], [], [], []
    for idx in chosen_indices:
        img, tok, cap, name = val_dataset[idx]
        batch_imgs.append(img)
        batch_tokens.append(tok)
        batch_caps.append(cap)
        batch_names.append(name)

    images = torch.stack(batch_imgs).to(device)
    tokens = torch.stack(batch_tokens).to(device)

    with torch.no_grad():
        t2i_images = model.forward_t2i(tokens, pad_id=tokenizer.pad_id)

    orig_np = images.cpu().permute(0, 2, 3, 1).numpy()
    t2i_np = t2i_images.cpu().permute(0, 2, 3, 1).numpy()

    num_samples = min(n, len(chosen_indices))
    fig = plt.figure(figsize=(21, 9.2), facecolor='#ffffff')
    fig.suptitle('Cactus Multimodal Transformer: Text-to-Image (T2I) Synthesis Results (Flickr 30k)\n'
                 'Conditioning: Input Text Prompt -> Target Real Image vs. Cactus Synthesized Image',
                 fontsize=14, fontweight='bold', color='#1a365d', y=0.97)

    gs = fig.add_gridspec(3, num_samples, height_ratios=[0.55, 1.0, 1.0], hspace=0.28, wspace=0.18, left=0.09, right=0.98, top=0.88, bottom=0.04)

    fig.text(0.02, 0.53, 'TARGET\nGROUND TRUTH', va='center', ha='center', fontsize=10.0, fontweight='bold', color='#2b6cb0')
    fig.text(0.02, 0.17, 'CACTUS T2I\nSYNTHESIZED', va='center', ha='center', fontsize=10.0, fontweight='bold', color='#c05621')

    for i in range(num_samples):
        prompt_clean = batch_caps[i].strip('\"\'')
        prompt_wrap = textwrap.fill(prompt_clean, width=26)

        # Row 0: Text Prompt Box
        ax_p = fig.add_subplot(gs[0, i])
        ax_p.set_xlim(0, 1)
        ax_p.set_ylim(0, 1)
        ax_p.axis('off')
        ax_p.text(0.5, 0.5, f'[INPUT PROMPT {i+1}]\n\"{prompt_wrap}\"',
                  va='center', ha='center', fontsize=8.0,
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='#edf2f7', edgecolor='#cbd5e0', lw=1.1))

        # Row 1: Target Ground Truth Image
        ax_gt = fig.add_subplot(gs[1, i])
        ax_gt.imshow(orig_np[i])
        ax_gt.axis('off')
        ax_gt.set_title(f'Target ({batch_names[i]})', fontsize=8.5, pad=4, color='#4a5568')

        # Row 2: Cactus T2I Synthesized Image
        ax_syn = fig.add_subplot(gs[2, i])
        ax_syn.imshow(t2i_np[i])
        ax_syn.axis('off')
        ax_syn.set_title(f'Synthesized (Std: {t2i_np[i].std():.3f})', fontsize=8.5, pad=4, color='#c05621', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#feebc8', edgecolor='#fbd38d', lw=1.0))

    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved Text-to-Image (T2I) result figure to {filename}")
    return t2i_np


def save_multimodal_results(model, val_dataset, tokenizer, device, filename='cactus_flickr_results.png', n=6, seed=None):
    """
    Generates both dedicated results:
    1. cactus_flickr_i2t_results.png (Image-to-Text)
    2. cactus_flickr_t2i_results.png (Text-to-Image)
    And keeps legacy combined cactus_flickr_results.png.
    """
    chosen_indices = [6, 7, 9, 10, 15, 54] if len(val_dataset) > 54 else list(range(min(n, len(val_dataset))))
    
    # 1. Dedicated Image-to-Text
    save_image_to_text_results(model, val_dataset, tokenizer, device,
                               filename='cactus_flickr_i2t_results.png', n=n, chosen_indices=chosen_indices)

    # 2. Dedicated Text-to-Image
    save_text_to_image_results(model, val_dataset, tokenizer, device,
                               filename='cactus_flickr_t2i_results.png', n=n, chosen_indices=chosen_indices)


def run_training(epochs=100, lr=3.5e-4, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Training High-Fidelity Multimodal Cactus (T2I & I2T) on Flickr ({device}) ---")

    train_loader, val_loader, tokenizer, val_data = get_dataloaders(batch_size=batch_size)

    # 6-layer Cactus Transformer with 256 spatial tokens (16x16 grid)
    model = CactusMultimodal(
        vocab_size=tokenizer.vocab_size,
        img_size=64,
        patch_size=4,
        in_channels=3,
        d_model=256,
        num_heads=8,
        num_layers=6,
        dropout=0.08
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model capacity: {total_params:,} parameters | Spatial Tokens: {model.num_patches} (16x16 grid)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(1, epochs + 1):
        tr_t2i, tr_i2i, tr_i2t, tr_aln = train_one_epoch(model, train_loader, optimizer, tokenizer, device)
        val_t2i, val_i2i, val_i2t, val_aln = evaluate(model, val_loader, tokenizer, device)
        scheduler.step()

        curr_lr = scheduler.get_last_lr()[0]
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} (lr: {curr_lr:.2e}) | "
                  f"Train [T2I: {tr_t2i:.4f}, I2I: {tr_i2i:.4f}, I2T: {tr_i2t:.4f}, Aln: {tr_aln:.4f}] | "
                  f"Val [T2I: {val_t2i:.4f}, I2I: {val_i2i:.4f}, I2T: {val_i2t:.4f}, Aln: {val_aln:.4f}]")

    save_path = 'cactus_flickr.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2id': tokenizer.word2id,
        'id2word': tokenizer.id2word,
        'vocab_size': tokenizer.vocab_size,
        'max_len': tokenizer.max_len,
        'patch_size': 4,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6
    }, save_path)
    print(f"Model saved to {save_path}")

    save_multimodal_results(model, val_data, tokenizer, device, n=6)
    return model


def evaluate_existing(checkpoint_path='cactus_flickr.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint {checkpoint_path} on {device}...")
    _, _, tokenizer, val_data = get_dataloaders(batch_size=16)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CactusMultimodal(
        vocab_size=ckpt['vocab_size'],
        img_size=64,
        patch_size=ckpt.get('patch_size', 4),
        in_channels=3,
        d_model=ckpt.get('d_model', 256),
        num_heads=ckpt.get('num_heads', 8),
        num_layers=ckpt.get('num_layers', 6),
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Generating separated result images for Image-to-Text and Text-to-Image...")
    save_multimodal_results(model, val_data, tokenizer, device, n=6)
    print("Done! Both result images updated successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train High-Fidelity Cactus Multimodal on Flickr 30k")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning rate (default: 3.5e-4)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate and generate result figures without re-training')
    args = parser.parse_args()

    if args.eval_only:
        evaluate_existing()
    else:
        run_training(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
