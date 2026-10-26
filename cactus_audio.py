import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from cactus import RMSNorm, GatedMHA, CactusEncoder, CactusDecoder
from cactus_multimodal import CactusMultimodal, ResBlock2d
from usm_conformer import USMAudioEmbeddingLayer


class ResBlock1d(nn.Module):
    # 1D residual block
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class AudioSynthesisHead(nn.Module):
    # Upsampling head for mel spectrogram prediction
    def __init__(self, d_model=256, n_mels=128):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        self.net = nn.Sequential(
            nn.Conv1d(d_model, 192, kernel_size=5, padding=2),
            nn.GroupNorm(8, 192),
            nn.SiLU(),
            ResBlock1d(192),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(192, 128, kernel_size=5, padding=2),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock1d(128),
            nn.Conv1d(128, n_mels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class CactusAudioMultimodal(nn.Module):
    # Tri-modal cactus with audio-to-text and text-to-audio
    def __init__(
        self,
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
        dropout=0.08
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_audio_tokens = num_audio_tokens
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # -------------------------------------------------------------
        # Core Cactus Multimodal Backbone (Text & Image)
        # -------------------------------------------------------------
        self.cactus_base = CactusMultimodal(
            vocab_size=vocab_size,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # -------------------------------------------------------------
        # Pretrained USM Audio Embedding Layer (2 Conv + 12 Conformer)
        # -------------------------------------------------------------
        self.audio_encoder = USMAudioEmbeddingLayer(
            d_model=d_model,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            win_length=800,
            hop_length=160,
            num_conformer_layers=num_conformer_layers,
            num_heads=4,
            ffn_dim=d_model * 4,
            depthwise_conv_kernel_size=31,
            dropout=dropout
        )

        # -------------------------------------------------------------
        # Audio Decoder (Official Cactus Transformer Architecture)
        # -------------------------------------------------------------
        self.audio_decoder = CactusDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_ffn=False
        )

        # Learnable temporal audio query seeds for audio decoding
        self.audio_queries = nn.Parameter(torch.randn(num_audio_tokens, 1, d_model) * 0.02)
        self.pos_embed_audio = nn.Parameter(torch.randn(num_audio_tokens, 1, d_model) * 0.02)

        # AdaMod: Semantic modulation for speech prosody, pitch, tempo from text embedding
        self.t2a_mod = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

        # Progressive 1D ResBlock Audio Synthesis Head
        self.audio_head = AudioSynthesisHead(d_model=d_model, n_mels=n_mels)

        # Vocoder for waveform synthesis
        self.inv_mel = T.InverseMelScale(n_stft=513, n_mels=n_mels, sample_rate=sample_rate)
        self.griffin_lim = T.GriffinLim(n_fft=1024, win_length=800, hop_length=160, power=1.0, n_iter=24)

    def load_pretrained_cactus_backbone(self, ckpt_path):
        """Loads pretrained weights from Question 4 cactus_flickr.pth into the base backbone."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict']
        self.cactus_base.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded pretrained Cactus backbone from {ckpt_path}")

    # =================================================================
    # AUDIO ENCODING
    # =================================================================
    def encode_audio(self, audio, lengths=None):
        """
        Passes raw audio waveform through USM 12-layer Conformer.
        Returns:
          audio_mem: (T_down, B, d_model)
          e_A: (B, d_model) pooled global audio embedding
          out_lengths: (B,)
        """
        return self.audio_encoder(audio, lengths=lengths)

    # =================================================================
    # AUDIO-TO-TEXT (A2T)
    # =================================================================
    def forward_a2t(self, audio, text_ids):
        """
        Audio-to-Text training forward pass.
        Cross-attends Cactus text decoder to USM 12-Conformer audio memory.
        Returns logits: (B, T, vocab_size)
        """
        audio_mem, e_A, _ = self.encode_audio(audio)
        full_audio_mem = torch.cat([e_A.unsqueeze(0), audio_mem], dim=0)

        # Embed text tokens: (B, seq_len) -> (T, B, d_model)
        x = self.cactus_base.text_embed(text_ids).permute(1, 0, 2)
        T_len = x.size(0)
        causal = torch.tril(torch.ones(T_len, T_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        decoded = self.cactus_base.text_decoder(x, full_audio_mem, mask=causal, modality='text')
        logits = self.cactus_base.text_head(decoded).permute(1, 0, 2)
        return logits

    def generate_text_from_audio(self, audio, tokenizer, max_len=28, beam_width=4, min_len=4):
        """
        Autoregressive transcription of speech audio into text with beam search.
        """
        self.eval()
        device = audio.device
        B = audio.size(0)

        with torch.no_grad():
            audio_mem, e_A, _ = self.encode_audio(audio)
            full_audio_mem = torch.cat([e_A.unsqueeze(0), audio_mem], dim=0)
            transcripts = []

            for b in range(B):
                single_mem = full_audio_mem[:, b:b+1]
                beams = [(0.0, [tokenizer.sos_id], False)]

                for step in range(max_len - 1):
                    all_candidates = []
                    any_active = False

                    for score, seq, finished in beams:
                        if finished:
                            all_candidates.append((score, seq, True))
                            continue

                        any_active = True
                        tokens_t = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(1)
                        x = self.cactus_base.text_embed(tokens_t)
                        T_len = x.size(0)
                        causal = torch.tril(torch.ones(T_len, T_len, device=device)).unsqueeze(0).unsqueeze(0)

                        decoded = self.cactus_base.text_decoder(x, single_mem, mask=causal, modality='text')
                        logits = self.cactus_base.text_head(decoded[-1, 0])

                        # Suppress special tokens
                        for bad_id in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.unk_id):
                            if bad_id < logits.size(-1):
                                logits[bad_id] = -1e9

                        # Penalize early eos
                        if len(seq) < min_len and tokenizer.eos_id < logits.size(-1):
                            logits[tokenizer.eos_id] -= 10.0

                        # Repetition penalty
                        repetition_penalty = 1.35
                        for prev_tok in set(seq):
                            if prev_tok != tokenizer.sos_id and prev_tok < logits.size(-1):
                                if logits[prev_tok] > 0:
                                    logits[prev_tok] /= repetition_penalty
                                else:
                                    logits[prev_tok] *= repetition_penalty

                        log_probs = F.log_softmax(logits, dim=-1)
                        topk_probs, topk_ids = torch.topk(log_probs, k=min(beam_width * 2, logits.size(-1)))

                        for k in range(topk_probs.size(0)):
                            tok = topk_ids[k].item()
                            lp = topk_probs[k].item()
                            all_candidates.append((score + lp, seq + [tok], tok == tokenizer.eos_id))

                    if not any_active:
                        break

                    all_candidates.sort(key=lambda c: c[0] / (len(c[1]) ** 0.65), reverse=True)
                    beams = all_candidates[:beam_width]

                best_seq = beams[0][1]
                transcripts.append(tokenizer.decode(best_seq))

        return transcripts

    # =================================================================
    # TEXT-TO-AUDIO (T2A)
    # =================================================================
    def forward_t2a(self, text_ids, pad_id=0):
        """
        Text-to-Audio forward pass:
        Text -> Cactus Text Encoder -> AdaMod Modulation -> Cactus Audio Decoder -> ResBlock Head -> Log-Mel Spectrogram.
        Returns:
          pred_mel: (B, n_mels, T_mel_frames)
          e_S: (B, d_model)
        """
        B = text_ids.size(0)
        text_mem, e_S = self.cactus_base.encode_text(text_ids, pad_id=pad_id)
        full_text_mem = torch.cat([e_S.unsqueeze(0), text_mem], dim=0)

        mem_mask = None
        if pad_id is not None:
            text_mask = (text_ids != pad_id)
            prefix_mask = torch.ones(B, 1, dtype=torch.bool, device=text_ids.device)
            full_mask = torch.cat([prefix_mask, text_mask], dim=1)
            mem_mask = full_mask.unsqueeze(1).unsqueeze(1)

        # Temporal audio query seeds + positional embeddings
        queries = self.audio_queries.expand(-1, B, -1) + self.pos_embed_audio

        # AdaMod prosody modulation from global text embedding e_S
        scale, shift = self.t2a_mod(e_S).chunk(2, dim=-1)
        x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)

        # Cactus Audio Decoder
        decoded = self.audio_decoder(x, full_text_mem, mem_mask=mem_mask, modality='audio')

        # Shape: (T_tokens, B, d_model) -> (B, d_model, T_tokens)
        latent_tokens = decoded.permute(1, 2, 0)

        # Progressive ResBlock1d Audio Synthesis Head
        pred_mel = self.audio_head(latent_tokens)  # (B, n_mels, 128)
        return pred_mel, e_S

    def generate_audio_from_text(self, text_ids, pad_id=0):
        """
        Synthesizes audible waveform from text IDs using Mel Head and Griffin-Lim Vocoder.
        Returns:
          waveform: (B, samples) at 16 kHz
          pred_mel: (B, n_mels, T_frames)
        """
        self.eval()
        with torch.no_grad():
            pred_mel, _ = self.forward_t2a(text_ids, pad_id=pad_id)
            # Reconstruct linear spectrogram and waveform
            linear_spec = self.inv_mel(torch.exp(pred_mel.clamp(max=5.0)))
            wav = self.griffin_lim(linear_spec)
        return wav, pred_mel

    # =================================================================
    # AUDIO-TO-AUDIO AUTOENCODING (A2A)
    # =================================================================
    def forward_a2a(self, audio):
        """Audio Autoencoding reconstruction."""
        B = audio.size(0)
        audio_mem, e_A, _ = self.encode_audio(audio)
        full_audio_mem = torch.cat([e_A.unsqueeze(0), audio_mem], dim=0)

        queries = self.audio_queries.expand(-1, B, -1) + self.pos_embed_audio
        scale, shift = self.t2a_mod(e_A).chunk(2, dim=-1)
        x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)

        decoded = self.audio_decoder(x, full_audio_mem, modality='audio')
        latent_tokens = decoded.permute(1, 2, 0)
        pred_mel = self.audio_head(latent_tokens)
        return pred_mel, e_A

    # =================================================================
    # EMERGENT CROSS-MODAL GENERATION (A2I & I2A)
    # =================================================================
    def forward_a2i(self, audio):
        """Emergent Audio-to-Image: Speech audio -> 64x64 RGB image."""
        self.eval()
        with torch.no_grad():
            B = audio.size(0)
            audio_mem, e_A, _ = self.encode_audio(audio)
            queries = self.cactus_base.spatial_queries.expand(-1, B, -1) + self.cactus_base.pos_embed_img
            scale, shift = self.cactus_base.t2i_mod(e_A).chunk(2, dim=-1)
            x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)
            decoded = self.cactus_base.image_decoder(x, audio_mem, modality='image')
            feat_map = decoded.permute(1, 2, 0).view(B, self.d_model, self.cactus_base.grid, self.cactus_base.grid)
            recon = self.cactus_base.image_head(feat_map)
            return torch.sigmoid(recon)

    def forward_i2a(self, images):
        """Emergent Image-to-Audio: Image -> Speech audio Mel Spectrogram."""
        self.eval()
        with torch.no_grad():
            B = images.size(0)
            img_mem, e_I = self.cactus_base.encode_image(images)
            queries = self.audio_queries.expand(-1, B, -1) + self.pos_embed_audio
            scale, shift = self.t2a_mod(e_I).chunk(2, dim=-1)
            x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)
            decoded = self.audio_decoder(x, img_mem, modality='audio')
            latent_tokens = decoded.permute(1, 2, 0)
            pred_mel = self.audio_head(latent_tokens)
            return pred_mel
