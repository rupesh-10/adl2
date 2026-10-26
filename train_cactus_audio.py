import os
import re
import json
import random
import argparse
import wave
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cactus_audio import CactusAudioMultimodal
from train_cactus_flickr import SimpleTokenizer


class PairedAudioTextDataset(Dataset):
    # Paired speech audio and text dataset
    def __init__(self, metadata_path='./data/audio/audio_captions.json', audio_dir='./data/audio',
                 tokenizer=None, max_audio_len=48000, target_mel_frames=128):
        with open(metadata_path) as f:
            self.items = json.load(f)
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len  # 3.0 seconds at 16 kHz
        self.target_mel_frames = target_mel_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav_file = os.path.join(self.audio_dir, item['audio'])

        # Read 16 kHz mono WAV with standard library wave
        with wave.open(wav_file, 'r') as wf:
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

        # Pad or crop audio to max_audio_len
        if len(audio) < self.max_audio_len:
            pad = np.zeros(self.max_audio_len - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        else:
            audio = audio[:self.max_audio_len]

        audio_t = torch.from_numpy(audio).float()
        token_ids = torch.tensor(self.tokenizer.encode(item['caption']), dtype=torch.long)

        return audio_t, token_ids, item['caption'], item['image']


def info_nce_loss(e_A, e_S, temperature=0.15):
    """Contrastive alignment between audio embedding e_A and text embedding e_S."""
    z_A = F.normalize(e_A, dim=-1)
    z_S = F.normalize(e_S, dim=-1)
    sim = torch.matmul(z_A, z_S.T) / temperature
    labels = torch.arange(e_A.size(0), device=e_A.device)
    loss_a2s = F.cross_entropy(sim, labels)
    loss_s2a = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss_a2s + loss_s2a)


def calc_spectral_loss(pred_mel, target_mel):
    """Multi-Scale Spectral Loss: L1 + Relative Spectral Convergence + Directional Cosine."""
    l1 = F.l1_loss(pred_mel, target_mel)
    spec_conv = torch.norm(target_mel - pred_mel) / (torch.norm(target_mel) + 1e-6)
    cosine_sim = F.cosine_similarity(pred_mel.flatten(1), target_mel.flatten(1), dim=-1).mean()
    cosine_loss = 1.0 - cosine_sim
    total = l1 + 0.5 * spec_conv + 0.25 * cosine_loss
    return total, l1, cosine_sim


def evaluate(model, val_loader, tokenizer, device, target_mel_frames=128):
    model.eval()
    total_loss = 0.0
    total_a2t_loss = 0.0
    total_t2a_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_cosine = 0.0
    count = 0

    with torch.no_grad():
        for audio, text_ids, captions, _ in val_loader:
            audio = audio.to(device)
            text_ids = text_ids.to(device)
            B = audio.size(0)

            # Target Log-Mel Spectrogram extraction
            target_mel = model.audio_encoder.extract_mel(audio).squeeze(1)  # (B, 128, T_raw)
            if target_mel.size(-1) != target_mel_frames:
                target_mel = F.interpolate(target_mel.unsqueeze(1), size=(128, target_mel_frames), mode='bilinear', align_corners=False).squeeze(1)

            # Forward Audio-to-Text
            logits_a2t = model.forward_a2t(audio, text_ids)
            loss_a2t = F.cross_entropy(
                logits_a2t[:, :-1].reshape(-1, model.vocab_size),
                text_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id
            )

            # A2T Accuracy calculation
            preds = logits_a2t[:, :-1].argmax(dim=-1)
            targets = text_ids[:, 1:]
            mask = (targets != tokenizer.pad_id)
            correct_tokens += ((preds == targets) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Forward Text-to-Audio
            pred_mel, e_S = model.forward_t2a(text_ids, pad_id=tokenizer.pad_id)
            if pred_mel.size(-1) != target_mel_frames:
                pred_mel = F.interpolate(pred_mel.unsqueeze(1), size=(128, target_mel_frames), mode='bilinear', align_corners=False).squeeze(1)

            loss_t2a, _, cos_sim = calc_spectral_loss(pred_mel, target_mel)

            # Multimodal InfoNCE
            _, e_A, _ = model.encode_audio(audio)
            loss_align = info_nce_loss(e_A, e_S)

            loss_combined = 1.2 * loss_a2t + 1.0 * loss_t2a + 0.3 * loss_align

            total_loss += loss_combined.item() * B
            total_a2t_loss += loss_a2t.item() * B
            total_t2a_loss += loss_t2a.item() * B
            total_cosine += cos_sim.item() * B
            count += B

    avg_loss = total_loss / count
    avg_a2t = total_a2t_loss / count
    avg_t2a = total_t2a_loss / count
    avg_cos = total_cosine / count
    accuracy = correct_tokens / max(1, total_tokens)

    return {
        'test_loss': avg_loss,
        'a2t_loss': avg_a2t,
        't2a_loss': avg_t2a,
        't2a_cosine': avg_cos,
        'a2t_accuracy': accuracy,
        'accuracy_pct': accuracy * 100.0
    }


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Flickr Captions to build or load tokenizer
    with open('./data/flickr30k/captions.json') as f:
        all_flickr = json.load(f)[:args.max_samples]
    captions = [p['caption'] for p in all_flickr]
    tokenizer = SimpleTokenizer(captions, max_vocab=3500, max_len=28)

    # Initialize Cactus Audio Multimodal Model
    model = CactusAudioMultimodal(
        vocab_size=3500,
        img_size=64,
        patch_size=4,
        in_channels=3,
        d_model=256,
        num_heads=8,
        num_layers=6,
        num_conformer_layers=12,
        num_audio_tokens=64,
        n_mels=128,
        sample_rate=16000,
        dropout=args.dropout
    ).to(device)

    # Load pretrained weights from Question 4 Cactus Flickr backbone if available
    if os.path.exists('cactus_flickr.pth'):
        try:
            model.load_pretrained_cactus_backbone('cactus_flickr.pth')
            print("Successfully transferred pretrained Cactus backbone weights!")
        except Exception as e:
            print(f"Notice on backbone load: {e}")

    # Save initial pretrained USM Conformer weights checkpoint
    os.makedirs('./data/audio', exist_ok=True)
    model.audio_encoder.save_pretrained('usm_conformer_pretrained.pth')

    # Prepare Dataset & DataLoaders
    full_dataset = PairedAudioTextDataset(
        metadata_path='./data/audio/audio_captions.json',
        audio_dir='./data/audio',
        tokenizer=tokenizer,
        max_audio_len=48000,
        target_mel_frames=128
    )

    total_len = len(full_dataset)
    train_size = int(args.train_ratio * total_len)  # 510 samples
    val_size = total_len - train_size               # 90 samples

    indices = list(range(total_len))
    rng = random.Random(42)
    rng.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_data = torch.utils.data.Subset(full_dataset, train_idx)
    val_data = torch.utils.data.Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset Split: {len(train_data)} Train Pairs, {len(val_data)} Test Pairs")

    # Parameter groups: Audio layers train at full lr, base backbone adapts at fine-tuning lr
    audio_params = list(model.audio_encoder.parameters()) + list(model.audio_decoder.parameters()) + \
                   list(model.audio_head.parameters()) + [model.audio_queries, model.pos_embed_audio] + \
                   list(model.t2a_mod.parameters())
    base_params = list(model.cactus_base.parameters())

    optimizer = torch.optim.AdamW([
        {'params': audio_params, 'lr': args.lr},
        {'params': base_params, 'lr': args.lr * 0.35}
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'a2t_loss': [], 't2a_loss': [], 'a2t_acc': [], 't2a_cos': []
    }

    best_loss = float('inf')

    print(f"\n{'='*75}")
    print(f"Beginning Joint Training: Multimodal Cactus with USM 12-Conformer Layers")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Learning Rate: {args.lr}")
    print(f"{'='*75}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_a2t = 0.0
        train_t2a = 0.0
        train_align = 0.0
        batches = 0

        for audio, text_ids, _, _ in train_loader:
            audio = audio.to(device)
            text_ids = text_ids.to(device)
            optimizer.zero_grad()

            # Target Mel Spectrogram
            target_mel = model.audio_encoder.extract_mel(audio).squeeze(1)
            if target_mel.size(-1) != 128:
                target_mel = F.interpolate(target_mel.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False).squeeze(1)

            # 1. Forward Audio-to-Text (A2T)
            logits_a2t = model.forward_a2t(audio, text_ids)
            loss_a2t = F.cross_entropy(
                logits_a2t[:, :-1].reshape(-1, model.vocab_size),
                text_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id
            )

            # 2. Forward Text-to-Audio (T2A)
            pred_mel, e_S = model.forward_t2a(text_ids, pad_id=tokenizer.pad_id)
            if pred_mel.size(-1) != 128:
                pred_mel = F.interpolate(pred_mel.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False).squeeze(1)

            loss_t2a, _, _ = calc_spectral_loss(pred_mel, target_mel)

            # 3. Audio-Text InfoNCE Contrastive Alignment
            _, e_A, _ = model.encode_audio(audio)
            loss_align = info_nce_loss(e_A, e_S)

            # Combined Loss
            loss = 1.2 * loss_a2t + 1.0 * loss_t2a + 0.3 * loss_align

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            train_loss += loss.item()
            train_a2t += loss_a2t.item()
            train_t2a += loss_t2a.item()
            train_align += loss_align.item()
            batches += 1

        scheduler.step()

        # Evaluate on validation set
        metrics = evaluate(model, val_loader, tokenizer, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss / batches)
        history['val_loss'].append(metrics['test_loss'])
        history['a2t_loss'].append(metrics['a2t_loss'])
        history['t2a_loss'].append(metrics['t2a_loss'])
        history['a2t_acc'].append(metrics['a2t_accuracy'])
        history['t2a_cos'].append(metrics['t2a_cosine'])

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Train Loss: {train_loss/batches:.4f} | "
                  f"Val Loss: {metrics['test_loss']:.4f} | "
                  f"A2T Acc: {metrics['accuracy_pct']:.2f}% | "
                  f"T2A CosSim: {metrics['t2a_cosine']:.4f}")

        if metrics['test_loss'] < best_loss:
            best_loss = metrics['test_loss']
            torch.save({
                'model_state_dict': model.state_dict(),
                'usm_conformer_state_dict': model.audio_encoder.state_dict(),
                'word2id': tokenizer.word2id,
                'id2word': tokenizer.id2word,
                'vocab_size': tokenizer.vocab_size,
                'max_len': tokenizer.max_len,
                'd_model': model.d_model,
                'num_heads': model.num_heads,
                'num_layers': model.num_layers,
                'num_conformer_layers': 12,
                'val_metrics': metrics,
                'epoch': epoch
            }, 'cactus_audio.pth')

    print(f"\nTraining completed! Best Validation Loss: {best_loss:.4f}")

    # Generate Evaluation Results and Plots
    generate_evaluation_artifacts(model, val_loader, tokenizer, history, device)


def generate_evaluation_artifacts(model, val_loader, tokenizer, history, device):
    """Generates qualitative predictions, synthesized samples, and visual plots."""
    print("Generating qualitative test predictions and result plots...")
    model.eval()

    # Get sample batch from val loader
    audio_batch, text_batch, captions, images = next(iter(val_loader))
    audio_batch = audio_batch.to(device)
    text_batch = text_batch.to(device)

    # 1. A2T Qualitative Transcription
    pred_transcripts = model.generate_text_from_audio(audio_batch[:4], tokenizer, beam_width=4)

    # 2. T2A Mel Spectrogram & Waveform Synthesis
    with torch.no_grad():
        target_mels = model.audio_encoder.extract_mel(audio_batch[:4]).squeeze(1)
        if target_mels.size(-1) != 128:
            target_mels = F.interpolate(target_mels.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False).squeeze(1)

        pred_mels, _ = model.forward_t2a(text_batch[:4], pad_id=tokenizer.pad_id)
        if pred_mels.size(-1) != 128:
            pred_mels = F.interpolate(pred_mels.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False).squeeze(1)

        # Synthesize audio waveforms via Griffin-Lim
        linear_spec = model.inv_mel(torch.exp(pred_mels.clamp(max=5.0)))
        synth_wavs = model.griffin_lim(linear_spec)

    # Save synthesized sample audio to disk
    sample_wav_path = "test_t2a_synthesized.wav"
    synth_np = synth_wavs[0].cpu().numpy()
    with wave.open(sample_wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        int_data = (np.clip(synth_np, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(int_data.tobytes())
    print(f"Sample synthesized speech saved to {sample_wav_path}")

    # Save transcription text comparison
    with open("test_a2t_predictions.txt", "w") as f:
        f.write("=== Question 6: Audio-to-Text (A2T) Speech Recognition Predictions ===\n\n")
        for i in range(min(4, len(captions))):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Ground Truth : {captions[i]}\n")
            f.write(f"  A2T Predicted: {pred_transcripts[i]}\n\n")

    # 3. Create Multi-Panel Visualization Plot
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.28)

    # --- ROW 1: Audio-to-Text Transcription & Audio Waveforms ---
    ax_wave1 = fig.add_subplot(gs[0, 0])
    raw_wav = audio_batch[0].cpu().numpy()
    t = np.linspace(0, len(raw_wav)/16000, len(raw_wav))
    ax_wave1.plot(t, raw_wav, color='#1f77b4', lw=0.8)
    ax_wave1.set_title("Input Speech Waveform (Sample 1)", fontsize=11, fontweight='bold')
    ax_wave1.set_xlabel("Time (seconds)", fontsize=9)
    ax_wave1.set_ylabel("Amplitude", fontsize=9)
    ax_wave1.grid(True, alpha=0.3)

    ax_mel_in = fig.add_subplot(gs[0, 1])
    im1 = ax_mel_in.imshow(target_mels[0].cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
    ax_mel_in.set_title("Input 128-Mel Spectrogram (USM Frontend)", fontsize=11, fontweight='bold')
    ax_mel_in.set_xlabel("Time Frames", fontsize=9)
    ax_mel_in.set_ylabel("Mel Frequency Bins", fontsize=9)
    plt.colorbar(im1, ax=ax_mel_in, fraction=0.046, pad=0.04)

    ax_text_box = fig.add_subplot(gs[0, 2])
    ax_text_box.axis('off')
    box_text = (
        "Audio-to-Text (A2T) Transcription:\n"
        "--------------------------------------------------\n"
        f"Ground Truth:\n  \"{captions[0][:48]}...\"\n\n"
        f"Conformer A2T Recognized:\n  \"{pred_transcripts[0]}\"\n\n"
        "--------------------------------------------------\n"
        f"Ground Truth (Sample 2):\n  \"{captions[1][:48]}...\"\n\n"
        f"Conformer A2T Recognized:\n  \"{pred_transcripts[1]}\""
    )
    ax_text_box.text(0.05, 0.5, box_text, fontsize=9.5, family='monospace', va='center',
                     bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', edgecolor='#2b579a', lw=1.5))

    # --- ROW 2: Text-to-Audio Synthesis (Ground Truth vs Model Synthesized) ---
    ax_gt_mel = fig.add_subplot(gs[1, 0])
    im2 = ax_gt_mel.imshow(target_mels[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax_gt_mel.set_title("Ground Truth Mel-Spectrogram", fontsize=11, fontweight='bold')
    ax_gt_mel.set_xlabel("Time Frames", fontsize=9)
    ax_gt_mel.set_ylabel("Mel Frequency Bins", fontsize=9)
    plt.colorbar(im2, ax=ax_gt_mel, fraction=0.046, pad=0.04)

    ax_pred_mel = fig.add_subplot(gs[1, 1])
    im3 = ax_pred_mel.imshow(pred_mels[0].cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    ax_pred_mel.set_title("Model Synthesized Mel-Spectrogram (T2A)", fontsize=11, fontweight='bold')
    ax_pred_mel.set_xlabel("Time Frames", fontsize=9)
    ax_pred_mel.set_ylabel("Mel Frequency Bins", fontsize=9)
    plt.colorbar(im3, ax=ax_pred_mel, fraction=0.046, pad=0.04)

    ax_synth_wave = fig.add_subplot(gs[1, 2])
    t_synth = np.linspace(0, len(synth_np)/16000, len(synth_np))
    ax_synth_wave.plot(t_synth, synth_np, color='#d62728', lw=0.8)
    ax_synth_wave.set_title("Synthesized Speech Waveform (Griffin-Lim)", fontsize=11, fontweight='bold')
    ax_synth_wave.set_xlabel("Time (seconds)", fontsize=9)
    ax_synth_wave.set_ylabel("Amplitude", fontsize=9)
    ax_synth_wave.grid(True, alpha=0.3)

    # --- ROW 3: Training Curves ---
    epochs = history['epoch']
    ax_loss = fig.add_subplot(gs[2, 0])
    ax_loss.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    ax_loss.plot(epochs, history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2, linestyle='--')
    ax_loss.set_title("Total Multimodal Loss Convergence", fontsize=11, fontweight='bold')
    ax_loss.set_xlabel("Epoch", fontsize=9)
    ax_loss.set_ylabel("Loss", fontsize=9)
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_a2t = fig.add_subplot(gs[2, 1])
    ax_a2t.plot(epochs, [a * 100 for a in history['a2t_acc']], color='#2ca02c', lw=2)
    ax_a2t.set_title("Audio-to-Text (A2T) Accuracy (%)", fontsize=11, fontweight='bold')
    ax_a2t.set_xlabel("Epoch", fontsize=9)
    ax_a2t.set_ylabel("Accuracy (%)", fontsize=9)
    ax_a2t.grid(True, alpha=0.3)

    ax_cos = fig.add_subplot(gs[2, 2])
    ax_cos.plot(epochs, history['t2a_cos'], color='#9467bd', lw=2)
    ax_cos.set_title("Text-to-Audio (T2A) Spectral Cosine Similarity", fontsize=11, fontweight='bold')
    ax_cos.set_xlabel("Epoch", fontsize=9)
    ax_cos.set_ylabel("Cosine Similarity", fontsize=9)
    ax_cos.grid(True, alpha=0.3)

    plt.suptitle("Question 6: Multimodal Cactus with USM 12-Conformer Audio Features (A2T & T2A)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig("cactus_audio_results.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved evaluation figure: cactus_audio_results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Multimodal Cactus with USM Audio Features")
    parser.add_argument('--max_samples', type=int, default=600, help="Max paired samples")
    parser.add_argument('--epochs', type=int, default=25, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--dropout', type=float, default=0.08, help="Dropout")
    parser.add_argument('--train_ratio', type=float, default=0.85, help="Train ratio")
    args = parser.parse_args()

    train(args)
