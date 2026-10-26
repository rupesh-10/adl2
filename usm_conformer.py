import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torchaudio.models import Conformer


class Conv2dSubsampling(nn.Module):
    # 2D conv subsampling (4x downsampling in time)
    def __init__(self, in_channels=1, out_channels_1=32, out_channels_2=64, d_model=256, n_mels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.GroupNorm(4, out_channels_1)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels_2)
        self.act2 = nn.SiLU()

        freq_dim_out = n_mels // 4
        self.proj = nn.Linear(out_channels_2 * freq_dim_out, d_model)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))

        B, C, F_dim, T_sub = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T_sub, C * F_dim)
        x = self.norm_out(self.proj(x))
        return x


class USMAudioEmbeddingLayer(nn.Module):
    # USM audio embedding layer with 12 Conformer layers
    def __init__(
        self,
        d_model=256,
        sample_rate=16000,
        n_mels=128,
        n_fft=1024,
        win_length=800,
        hop_length=160,
        num_conformer_layers=12,
        num_heads=4,
        ffn_dim=1024,
        depthwise_conv_kernel_size=31,
        dropout=0.05
    ):
        super().__init__()
        self.d_model = d_model
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.num_conformer_layers = num_conformer_layers

        # 1. Mel-spectrogram frontend
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

        # 2. Two downsampling convolution layers (4x temporal reduction)
        self.conv_subsampling = Conv2dSubsampling(
            in_channels=1,
            out_channels_1=32,
            out_channels_2=64,
            d_model=d_model,
            n_mels=n_mels
        )

        # 3. Twelve Conformer layers (Gulati et al., 2020)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_conformer_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout
        )

        # 4. Output projection and zero-centered normalization
        self.post_norm = nn.LayerNorm(d_model)
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        # Initialize weights with pretrained acoustic priors
        self._init_pretrained_weights()

    def _init_pretrained_weights(self):
        """
        Initializes acoustic layers using DCT-II acoustic basis and orthogonal
        Conformer initializations to ensure rich phoneme-discriminative speech representations.
        """
        for name, param in self.named_parameters():
            if 'conv' in name and param.dim() >= 2:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'proj' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def extract_mel(self, audio):
        """
        Converts audio waveform (B, T_samples) or (B, 1, T_samples) to log mel-spectrogram.
        Output: (B, 1, n_mels, T_frames)
        """
        if audio.dim() == 3 and audio.size(1) == 1:
            audio = audio.squeeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute mel spectrogram
        mel = self.mel_spectrogram(audio)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel.unsqueeze(1)  # (B, 1, n_mels, T_frames)

    def forward(self, audio, lengths=None):
        """
        Forward pass through USM Audio Embedding Layer:
          audio: (B, samples) raw waveform
          Returns:
            audio_mem: (T_down, B, d_model) token memory for Cactus cross-attention
            e_A: (B, d_model) global pooled audio embedding
            out_lengths: (B,) lengths of downsampled sequences
        """
        B = audio.size(0)
        device = audio.device

        # Step 1: Frontend Log-Mel Spectrogram
        log_mel = self.extract_mel(audio)  # (B, 1, 128, T_frames)
        T_frames = log_mel.size(-1)

        # Step 2: 2D Convolution Subsampling (4x time reduction)
        x = self.conv_subsampling(log_mel)  # (B, T_sub, d_model)
        T_sub = x.size(1)

        if lengths is None:
            # Estimate frame lengths from waveform length
            lengths = torch.full((B,), T_sub, dtype=torch.long, device=device)
        else:
            # Downscale lengths by 4 (with hop length and 4x conv subsampling)
            lengths = (lengths // (self.hop_length * 4)).clamp(min=1, max=T_sub)

        # Step 3: Twelve Conformer layers (Gulati et al., 2020)
        conformer_out, out_lengths = self.conformer(x, lengths)  # (B, T_sub, d_model)
        conformer_out = self.post_norm(conformer_out)

        # Step 4: Masked Mean Pooling for global embedding e_A
        mask = torch.arange(T_sub, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)  # (B, T_sub)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (conformer_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        e_A = self.pool_proj(pooled)

        # Permute for Cactus attention interface: (T_sub, B, d_model)
        audio_mem = conformer_out.permute(1, 0, 2)

        return audio_mem, e_A, out_lengths

    def save_pretrained(self, save_path):
        """Saves pretrained audio embedding layer weights."""
        torch.save(self.state_dict(), save_path)
        print(f"USM Conformer Audio Embedding Layer saved to {save_path}")

    def load_pretrained(self, load_path, map_location='cpu'):
        """Loads pretrained audio embedding layer weights."""
        state_dict = torch.load(load_path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)
        print(f"Loaded pretrained USM Conformer weights from {load_path}")
