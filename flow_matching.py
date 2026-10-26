import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cactus import RMSNorm, GatedMHA
from cactus_multimodal import SwiGLU


class TimeEmbedding(nn.Module):
    # Sinusoidal time embedding
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device, dtype=torch.float32) * (math.log(10000.0) / max(half - 1, 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return self.mlp(emb)


class FlowBlock(nn.Module):
    # Transformer block conditioned on time and guide tokens
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.time_mod1 = nn.Linear(d_model, d_model * 2)
        self.cross_attn = GatedMHA(d_model, num_heads)

        self.norm2 = RMSNorm(d_model)
        self.time_mod2 = nn.Linear(d_model, d_model * 2)
        self.ffn = SwiGLU(d_model)

    def forward(self, x, t_emb, guide_tokens):
        # 1. Multi-head Cross-Attention attending to structured e_S guide tokens
        scale1, shift1 = self.time_mod1(t_emb).chunk(2, dim=-1)
        h = self.norm1(x) * (1.0 + torch.tanh(scale1)) + torch.tanh(shift1)

        q = h.unsqueeze(0)        # (1, B, d_model)
        attn_out = self.cross_attn(q, guide_tokens, guide_tokens).squeeze(0)
        x = x + attn_out

        # 2. Gated SwiGLU FFN with AdaLN time conditioning
        scale2, shift2 = self.time_mod2(t_emb).chunk(2, dim=-1)
        h2 = self.norm2(x) * (1.0 + torch.tanh(scale2)) + torch.tanh(shift2)
        x = x + self.ffn(h2)
        return x


class FlowMatchingDecoder(nn.Module):
    """
    Deep Flow Matching Decoder:
    Projects guide sentence embedding e_S into 8 structured guide tokens so that
    cross-attention dynamically attends over diverse semantic aspects of e_S.
    Predicts the clean target embedding x_1 directly to ensure constant-gradient
    supervision across all noise levels t in [0, 1].
    """
    def __init__(self, d_model=256, num_heads=8, num_layers=6, num_guide_tokens=8):
        super().__init__()
        self.d_model = d_model
        self.num_guide_tokens = num_guide_tokens
        self.time_emb = TimeEmbedding(d_model)

        # Multi-aspect guide projection: expands global e_S into 8 distinct guide tokens
        self.guide_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, num_guide_tokens * d_model)
        )
        self.null_guide = nn.Parameter(torch.zeros(1, d_model))

        self.in_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList([
            FlowBlock(d_model=d_model, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.out_norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_t, t, e_S):
        B = x_t.size(0)
        t_emb = self.time_emb(t)

        # Project e_S into (num_guide_tokens, B, d_model) for genuine cross-attention
        guide_tokens = self.guide_proj(e_S).view(B, self.num_guide_tokens, self.d_model).permute(1, 0, 2)

        h = self.in_proj(x_t)
        for block in self.blocks:
            h = block(h, t_emb, guide_tokens)

        # Predict clean target representation x_1
        pred_x1 = self.out_proj(self.out_norm(h))
        return pred_x1


class FlowMatching(nn.Module):
    """
    Target-Supervised Conditional Flow Matching:
    - Optimal Transport Path: x_t = (1 - t) * x_0 + t * x_1
    - Implied Velocity Field: v_theta(x_t, t, e_S) = (pred_x1 - x_t) / (1 - t)
    - Directly optimizes target prediction to achieve high cosine similarity
      and low MSE across all noise levels t in [0, 1].
    """
    def __init__(self, d_model=256, num_heads=8, num_layers=6, num_guide_tokens=8):
        super().__init__()
        self.d_model = d_model
        self.decoder = FlowMatchingDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            num_guide_tokens=num_guide_tokens
        )

    def forward_noise(self, x_1, t, x_0=None):
        """Creates noisy version x_t of embedding x_1 at time t (Optimal Transport path)."""
        if x_0 is None:
            x_0 = torch.randn_like(x_1)

        t_exp = t.view(-1, 1)
        x_t = (1.0 - t_exp) * x_0 + t_exp * x_1
        target_velocity = x_1 - x_0
        return x_t, target_velocity, x_0

    def compute_loss(self, e_I, e_S, drop_prob=0.15):
        """
        Loss combining direct target reconstruction MSE, directional cosine alignment,
        and velocity field consistency across continuous time t in [0, 1].
        """
        B = e_I.size(0)
        t = torch.rand(B, device=e_I.device)
        x_t, target_v, x_0 = self.forward_noise(e_I, t)

        # Classifier-free guidance dropout during training
        if drop_prob > 0.0 and self.training:
            mask = torch.rand(B, 1, device=e_I.device) < drop_prob
            e_S_cond = torch.where(mask, self.decoder.null_guide.expand(B, -1), e_S)
        else:
            e_S_cond = e_S

        pred_x1 = self.decoder(x_t, t, e_S_cond)

        # 1. Direct Target Reconstruction Loss (Equal gradient across all t)
        mse_target = F.mse_loss(pred_x1, e_I)
        cosine_sim = F.cosine_similarity(pred_x1, e_I, dim=-1).mean()
        cos_loss = 1.0 - cosine_sim

        # 2. Implied Velocity Field Loss
        t_safe = torch.clamp(t.view(-1, 1), max=0.99)
        implied_v = (pred_x1 - x_t) / (1.0 - t_safe + 1e-4)
        mse_vel = F.mse_loss(implied_v, target_v)

        total_loss = mse_target + 0.8 * cos_loss + 0.1 * torch.clamp(mse_vel, max=10.0)
        return total_loss, mse_target.item(), cosine_sim.item()

    @torch.no_grad()
    def get_velocity(self, x, t_curr, e_S, cfg_scale=1.5):
        """Calculates conditional flow velocity dx/dt with Classifier-Free Guidance."""
        B = x.size(0)
        t_tensor = torch.full((B,), t_curr, device=x.device, dtype=torch.float32)

        pred_cond = self.decoder(x, t_tensor, e_S)

        if cfg_scale > 1.0:
            null_g = self.decoder.null_guide.expand(B, -1)
            pred_uncond = self.decoder(x, t_tensor, null_g)
            pred_x1 = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred_x1 = pred_cond

        t_safe = min(t_curr, 0.999)
        v = (pred_x1 - x) / (1.0 - t_safe + 1e-4)
        return v, pred_x1

    @torch.no_grad()
    def recreate_from_noise(self, noisy_e, t_start, e_S, steps=50, method='rk2', cfg_scale=1.5):
        """
        Recreates target embedding e_1 from noisy embedding noisy_e (starting at t=t_start)
        using sentence embedding e_S as guide via ODE integration from t_start to 1.0.
        """
        self.eval()
        B = noisy_e.size(0)
        x = noisy_e.clone()
        dt = (1.0 - t_start) / max(steps, 1)

        trajectory = [(t_start, x.clone())]

        for step in range(steps):
            t_curr = t_start + step * dt

            if method == 'rk2' and step < steps - 1:
                # 2nd-order Runge-Kutta (Midpoint Method)
                v1, _ = self.get_velocity(x, t_curr, e_S, cfg_scale=cfg_scale)
                t_mid = t_curr + 0.5 * dt
                x_mid = x + 0.5 * dt * v1
                v2, _ = self.get_velocity(x_mid, t_mid, e_S, cfg_scale=cfg_scale)
                x = x + dt * v2
            else:
                v, _ = self.get_velocity(x, t_curr, e_S, cfg_scale=cfg_scale)
                x = x + dt * v

            trajectory.append((t_curr + dt, x.clone()))

        # Direct high-fidelity target refinement at t=1.0
        _, final_pred = self.get_velocity(x, 1.0, e_S, cfg_scale=cfg_scale)
        final_x = 0.5 * x + 0.5 * final_pred

        return final_x, trajectory

    @torch.no_grad()
    def sample(self, e_S, steps=50, method='rk2', cfg_scale=1.5):
        """Generates embedding from pure noise t=0 to clean embedding t=1 guided by e_S."""
        B = e_S.size(0)
        x_0 = torch.randn(B, self.d_model, device=e_S.device)
        return self.recreate_from_noise(x_0, t_start=0.0, e_S=e_S, steps=steps, method=method, cfg_scale=cfg_scale)
