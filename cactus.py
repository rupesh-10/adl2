import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZCRMSNorm(nn.Module):
    # Zero-centered RMSNorm
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        x_centered = x - x.mean(dim=-1, keepdim=True)
        rms = x_centered.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x_centered / rms)


# Backward-compatible alias
RMSNorm = ZCRMSNorm


def precompute_rope(head_dim, max_len=512, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[:x.size(0)].unsqueeze(1).unsqueeze(1)
    s = sin[:x.size(0)].unsqueeze(1).unsqueeze(1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class GatedMHA(nn.Module):
    """
    Grouped Query Attention (GQA) with RoPE and learnable Gated Output.
    Supports causal masking and cross-attention.
    """
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.05, max_len=512):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.kvh = num_kv_heads or num_heads
        self.hd = d_model // num_heads
        self.d = d_model

        self.Wq = nn.Linear(d_model, num_heads * self.hd)
        self.Wk = nn.Linear(d_model, self.kvh * self.hd)
        self.Wv = nn.Linear(d_model, self.kvh * self.hd)
        self.Wo = nn.Linear(d_model, d_model)

        self.gate = nn.Parameter(torch.ones(d_model))
        self.drop = nn.Dropout(dropout)

        cos, sin = precompute_rope(self.hd, max_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, q, k, v, mask=None, use_rope=True):
        Tq, B, _ = q.shape
        Tk = k.size(0)

        Q = self.Wq(q).view(Tq, B, self.h, self.hd)
        K = self.Wk(k).view(Tk, B, self.kvh, self.hd)
        V = self.Wv(v).view(Tk, B, self.kvh, self.hd)

        if use_rope:
            cos = self.rope_cos.to(Q.dtype)
            sin = self.rope_sin.to(Q.dtype)
            Q = apply_rope(Q, cos, sin)
            K = apply_rope(K, cos, sin)

        if self.kvh != self.h:
            repeat = self.h // self.kvh
            K = K.repeat_interleave(repeat, dim=2)
            V = V.repeat_interleave(repeat, dim=2)

        Q = Q.permute(1, 2, 0, 3)
        K = K.permute(1, 2, 0, 3)
        V = V.permute(1, 2, 0, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hd)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = self.drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).permute(2, 0, 1, 3).contiguous().view(Tq, B, self.d)
        return torch.sigmoid(self.gate) * self.Wo(out)


class ExpertFFN(nn.Module):
    """Gated SwiGLU Feed-Forward Expert Network."""
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(d_model * 8 / 3)
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class ModalityExperts(nn.Module):
    """
    [NEW] Modality experts: one feed-forward network runs per layer.
    Routes between Text expert feed-forward and Image expert feed-forward.
    """
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        self.text_expert = ExpertFFN(d_model, hidden_dim)
        self.image_expert = ExpertFFN(d_model, hidden_dim)

    def forward(self, x, modality='text'):
        if modality == 'image':
            return self.image_expert(x)
        else:
            return self.text_expert(x)


class CactusEncoderLayer(nn.Module):
    """
    Official Cactus/Chameleon Encoder Layer:
    ZCRMSNorm -> Grouped Query Self-Attention RoPE -> Gated Residual (+)
    Optional Modality Experts: ZCRMSNorm -> Modality Experts (Text or Image FFN) -> Gated Residual (+)
    """
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.05, use_ffn=True):
        super().__init__()
        self.use_ffn = use_ffn
        if use_ffn:
            # 1. Self-Attention Block
            self.norm1 = ZCRMSNorm(d_model)
            self.attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)
            self.attn_gate = nn.Parameter(torch.zeros(d_model))

            # 2. Modality Experts Feed-Forward Block
            self.norm2 = ZCRMSNorm(d_model)
            self.modality_experts = ModalityExperts(d_model)
            self.ffn_gate = nn.Parameter(torch.zeros(d_model))
            self.drop = nn.Dropout(dropout)
        else:
            self.norm = ZCRMSNorm(d_model)
            self.attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)

    def forward(self, x, modality='text'):
        if self.use_ffn:
            # 1. GQA Self-Attention with RoPE + Gated Residual (+)
            n1 = self.norm1(x)
            attn_out = self.attn(n1, n1, n1, use_rope=True)
            x = x + torch.tanh(self.attn_gate) * self.drop(attn_out)

            # 2. Modality Expert Feed-Forward + Gated Residual (+)
            n2 = self.norm2(x)
            ffn_out = self.modality_experts(n2, modality=modality)
            x = x + torch.tanh(self.ffn_gate) * self.drop(ffn_out)
            return x
        else:
            return x + self.attn(self.norm(x), self.norm(x), self.norm(x), use_rope=True)


class CactusEncoder(nn.Module):
    """
    Official Cactus Encoder Stack with Pre-ZCRMSNorm, optional Modality Experts,
    and Post-ZCRMSNorm before outputting KV to decoder.
    """
    def __init__(self, d_model, num_heads, num_layers=12, num_kv_heads=None, dropout=0.05, use_ffn=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CactusEncoderLayer(d_model, num_heads, num_kv_heads, dropout, use_ffn=use_ffn)
            for _ in range(num_layers)
        ])
        self.norm = ZCRMSNorm(d_model)

    def forward(self, x, modality='text'):
        for layer in self.layers:
            x = layer(x, modality=modality)
        return self.norm(x)


class CactusDecoderLayer(nn.Module):
    """
    Official Cactus/Chameleon Decoder Layer:
    ZCRMSNorm -> GQA Causal Self-Attention RoPE -> Gated Residual (+) ->
    ZCRMSNorm -> GQA Cross-Attention (Q: Dec, KV: Enc) -> Gated Residual (+) ->
    Optional Modality Experts: ZCRMSNorm -> Modality Experts (Text or Image FFN) -> Gated Residual (+)
    """
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.05, use_ffn=True):
        super().__init__()
        self.use_ffn = use_ffn
        if use_ffn:
            # 1. Causal Self-Attention
            self.norm1 = ZCRMSNorm(d_model)
            self.self_attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)
            self.self_gate = nn.Parameter(torch.zeros(d_model))

            # 2. Cross-Attention
            self.norm2 = ZCRMSNorm(d_model)
            self.cross_attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)
            self.cross_gate = nn.Parameter(torch.zeros(d_model))

            # 3. Modality Experts
            self.norm3 = ZCRMSNorm(d_model)
            self.modality_experts = ModalityExperts(d_model)
            self.ffn_gate = nn.Parameter(torch.zeros(d_model))
            self.drop = nn.Dropout(dropout)
        else:
            self.norm1 = ZCRMSNorm(d_model)
            self.self_attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)
            self.norm2 = ZCRMSNorm(d_model)
            self.cross_attn = GatedMHA(d_model, num_heads, num_kv_heads, dropout)

    def forward(self, x, memory, mask=None, mem_mask=None, modality='text'):
        if self.use_ffn:
            # 1. GQA Causal Self-Attention with RoPE + Gated Residual (+)
            n1 = self.norm1(x)
            s_out = self.self_attn(n1, n1, n1, mask=mask, use_rope=True)
            x = x + torch.tanh(self.self_gate) * self.drop(s_out)

            # 2. GQA Cross-Attention (Q: Decoder, KV: Encoder) + Gated Residual (+)
            if memory is not None:
                n2 = self.norm2(x)
                c_out = self.cross_attn(n2, memory, memory, mask=mem_mask, use_rope=False)
                x = x + torch.tanh(self.cross_gate) * self.drop(c_out)

            # 3. Modality Expert Feed-Forward + Gated Residual (+)
            n3 = self.norm3(x)
            ffn_out = self.modality_experts(n3, modality=modality)
            x = x + torch.tanh(self.ffn_gate) * self.drop(ffn_out)
            return x
        else:
            n1 = self.norm1(x)
            x = x + self.self_attn(n1, n1, n1, mask=mask, use_rope=True)
            if memory is not None:
                n2 = self.norm2(x)
                x = x + self.cross_attn(n2, memory, memory, mask=mem_mask, use_rope=False)
            return x


class CactusDecoder(nn.Module):
    """
    Official Cactus Decoder Stack with Pre-ZCRMSNorm, Causal GQA, Cross-GQA,
    optional Modality Experts, and Post-ZCRMSNorm before prediction head.
    """
    def __init__(self, d_model, num_heads, num_layers=8, num_kv_heads=None, dropout=0.05, encoder_to='all', use_ffn=True):
        super().__init__()
        self.layers = nn.ModuleList([
            CactusDecoderLayer(d_model, num_heads, num_kv_heads, dropout, use_ffn=use_ffn)
            for _ in range(num_layers)
        ])
        self.encoder_to = encoder_to
        self.norm = ZCRMSNorm(d_model)

    def forward(self, x, memory, mask=None, mem_mask=None, modality='text'):
        for i, layer in enumerate(self.layers):
            mem = memory if (self.encoder_to == 'all' or i == 0) else None
            x = layer(x, mem, mask=mask, mem_mask=mem_mask, modality=modality)
        return self.norm(x)


class CactusImageAutoencoder(nn.Module):
    """
    Cactus Image Autoencoder utilizing CactusEncoder and CactusDecoder
    with Image Modality Experts and ZCRMSNorm.
    """
    def __init__(self, img_size=64, patch_size=8, in_channels=3,
                 d_model=128, num_heads=4, num_layers=3, dropout=0.05):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.queries = nn.Parameter(torch.randn(self.num_patches, 1, d_model) * 0.02)

        self.encoder = CactusEncoder(d_model, num_heads, num_layers, dropout=dropout)
        self.decoder = CactusDecoder(d_model, num_heads, num_layers, dropout=dropout)

        self.head = nn.Linear(d_model, self.patch_dim)

    def forward(self, x):
        B = x.size(0)
        p, g = self.patch_size, self.grid

        # Patchify
        patches = x.view(B, self.in_channels, g, p, g, p).permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, self.num_patches, self.patch_dim)

        tokens = self.patch_proj(patches).permute(1, 0, 2)
        memory = self.encoder(tokens, modality='image')

        q = self.queries.expand(-1, B, -1)
        decoded = self.decoder(q, memory, modality='image')

        recon_patches = self.head(decoded).permute(1, 0, 2)
        recon = recon_patches.view(B, g, g, self.in_channels, p, p).permute(0, 3, 1, 4, 2, 5).contiguous()
        recon = recon.view(B, self.in_channels, self.img_size, self.img_size)

        return torch.sigmoid(recon)
