import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cactus import RMSNorm, GatedMHA, CactusEncoder, CactusDecoder


class ResBlock2d(nn.Module):
    # 2D residual block
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels)
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class CactusMultimodal(nn.Module):
    # Multimodal transformer for image-text generation
    def __init__(self, vocab_size, img_size=64, patch_size=4, in_channels=3,
                 d_model=256, num_heads=8, num_layers=6, dropout=0.08):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = img_size // patch_size  # 16
        self.num_patches = self.grid * self.grid  # 256
        self.patch_dim = in_channels * patch_size * patch_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token & Image Embeddings
        self.text_embed = nn.Embedding(vocab_size, d_model)

        # Overlapping convolutional image stem for continuous spatial representations
        self.image_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU()
        )

        # 2D Positional Embeddings for image grid
        self.pos_embed_img = nn.Parameter(torch.randn(self.num_patches, 1, d_model) * 0.02)

        # Learnable spatial query seeds for decoder
        self.spatial_queries = nn.Parameter(torch.randn(self.num_patches, 1, d_model) * 0.02)

        # Encoders (Official Cactus Architecture: Pre-ZCRMSNorm + GQA RoPE + Gated Residual, NO FFN)
        self.text_encoder = CactusEncoder(d_model, num_heads, num_layers, dropout=dropout, use_ffn=False)
        self.image_encoder = CactusEncoder(d_model, num_heads, num_layers, dropout=dropout, use_ffn=False)

        # Decoders (Official Cactus Architecture: Pre-ZCRMSNorm + Causal GQA RoPE + Gated Residual + Cross-GQA + Gated Residual, NO FFN)
        self.image_decoder = CactusDecoder(d_model, num_heads, num_layers, dropout=dropout, use_ffn=False)
        self.text_decoder = CactusDecoder(d_model, num_heads, num_layers, dropout=dropout, use_ffn=False)

        # AdaMod: Semantic color, lighting & style modulation from global embedding
        self.t2i_mod = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2)
        )

        # High-Fidelity Progressive Upsampling Image Head with Residual Blocks
        self.image_head = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x16 -> 32x32
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x32 -> 64x64
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResBlock2d(32),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
        )

        # Text Prediction Head
        self.text_head = nn.Linear(d_model, vocab_size, bias=True)

    def encode_text(self, text_ids, pad_id=0):
        # text_ids: (B, seq_len) -> (T, B, d_model)
        x = self.text_embed(text_ids).permute(1, 0, 2)
        mem = self.text_encoder(x, modality='text')
        if pad_id is not None:
            mask = (text_ids != pad_id).float().unsqueeze(-1)
            mask_t = mask.permute(1, 0, 2)
            e_S = (mem * mask_t).sum(dim=0) / mask.sum(dim=1).clamp(min=1.0)
        else:
            e_S = mem.mean(dim=0)
        return mem, e_S

    def encode_image(self, images):
        # images: (B, C, 64, 64) -> stem -> (B, d_model, 16, 16) -> (256, B, d_model)
        B = images.size(0)
        feat = self.image_stem(images)
        x = feat.flatten(2).permute(2, 0, 1) + self.pos_embed_img
        mem = self.image_encoder(x, modality='image')
        e_I = mem.mean(dim=0)
        return mem, e_I

    def forward_t2i(self, text_ids, pad_id=0):
        # Text to image generation
        B = text_ids.size(0)
        text_mem, e_S = self.encode_text(text_ids, pad_id=pad_id)

        # Prefix sentence embedding to text memory
        full_text_mem = torch.cat([e_S.unsqueeze(0), text_mem], dim=0)

        mem_mask = None
        if pad_id is not None:
            text_mask = (text_ids != pad_id)
            prefix_mask = torch.ones(B, 1, dtype=torch.bool, device=text_ids.device)
            full_mask = torch.cat([prefix_mask, text_mask], dim=1)
            mem_mask = full_mask.unsqueeze(1).unsqueeze(1)

        # Spatial queries with pos embeddings
        queries = self.spatial_queries.expand(-1, B, -1) + self.pos_embed_img

        # Modulate queries via e_S
        scale, shift = self.t2i_mod(e_S).chunk(2, dim=-1)
        x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)

        # Image decoder cross-attention
        decoded = self.image_decoder(x, full_text_mem, mem_mask=mem_mask, modality='image')

        # Reshape to feature map
        feat_map = decoded.permute(1, 2, 0).view(B, self.d_model, self.grid, self.grid)

        # Upsample to 64x64
        recon_images = self.image_head(feat_map)
        return torch.sigmoid(recon_images)

    def forward_i2i(self, images):
        # Image autoencoding
        B = images.size(0)
        img_mem, e_I = self.encode_image(images)

        queries = self.spatial_queries.expand(-1, B, -1) + self.pos_embed_img
        scale, shift = self.t2i_mod(e_I).chunk(2, dim=-1)
        x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)

        decoded = self.image_decoder(x, img_mem, modality='image')
        feat_map = decoded.permute(1, 2, 0).view(B, self.d_model, self.grid, self.grid)
        recon_images = self.image_head(feat_map)
        return torch.sigmoid(recon_images)

    def decode_from_embedding(self, emb):
        # Decode latent vector into image
        self.eval()
        with torch.no_grad():
            B = emb.size(0)
            mem = emb.unsqueeze(0)
            queries = self.spatial_queries.expand(-1, B, -1) + self.pos_embed_img
            scale, shift = self.t2i_mod(emb).chunk(2, dim=-1)
            x = queries * (1.0 + 0.3 * torch.tanh(scale).unsqueeze(0)) + 0.3 * torch.tanh(shift).unsqueeze(0)
            decoded = self.image_decoder(x, mem, modality='image')
            feat_map = decoded.permute(1, 2, 0).view(B, self.d_model, self.grid, self.grid)
            recon = self.image_head(feat_map)
            return torch.sigmoid(recon)

    def forward_i2t(self, images, text_ids):
        # Image to text caption prediction
        img_mem, _ = self.encode_image(images)

        x = self.text_embed(text_ids).permute(1, 0, 2)
        T = x.size(0)
        causal = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        decoded = self.text_decoder(x, img_mem, mask=causal, modality='text')
        logits = self.text_head(decoded).permute(1, 0, 2)
        return logits

    def generate_caption(self, images, tokenizer, max_len=28, repetition_penalty=1.4, beam_width=4, min_len=6):
        # Autoregressive caption generation with beam search
        self.eval()
        device = images.device
        B = images.size(0)

        with torch.no_grad():
            img_mem, _ = self.encode_image(images)
            captions = []

            for b in range(B):
                single_mem = img_mem[:, b:b+1]

                # Beam state: (score, token_sequence, finished_boolean)
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
                        x = self.text_embed(tokens_t)
                        T = x.size(0)
                        causal = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

                        decoded = self.text_decoder(x, single_mem, mask=causal, modality='text')
                        logits = self.text_head(decoded[-1, 0])

                        # Suppress special tokens
                        for bad_id in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.unk_id):
                            if bad_id < logits.size(-1):
                                logits[bad_id] = -1e9

                        # Penalize premature <eos>
                        if len(seq) < min_len:
                            if tokenizer.eos_id < logits.size(-1):
                                logits[tokenizer.eos_id] -= 10.0

                        # Repetition penalty
                        if repetition_penalty > 1.0:
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
                            new_score = score + lp
                            new_seq = seq + [tok]
                            new_finished = (tok == tokenizer.eos_id)
                            all_candidates.append((new_score, new_seq, new_finished))

                    if not any_active:
                        break

                    all_candidates.sort(key=lambda c: c[0] / (len(c[1]) ** 0.65), reverse=True)
                    beams = all_candidates[:beam_width]

                best_seq = beams[0][1]
                captions.append(tokenizer.decode(best_seq))

        return captions


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(d_model * 8 / 3)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
