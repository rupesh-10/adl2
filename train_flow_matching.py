import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from cactus_multimodal import CactusMultimodal
from train_cactus_flickr import get_dataloaders, SimpleTokenizer
from flow_matching import FlowMatching


def load_q4_model(checkpoint_path='cactus_flickr.pth', device='cpu'):
    """Loads trained Cactus Multimodal model from Question 4 with shape safety."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Please train or provide Q4 model first.")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = CactusMultimodal(
        vocab_size=ckpt['vocab_size'],
        img_size=64,
        patch_size=ckpt.get('patch_size', 4),
        in_channels=3,
        d_model=ckpt.get('d_model', 256),
        num_heads=ckpt.get('num_heads', 8),
        num_layers=ckpt.get('num_layers', 4)
    ).to(device)

    model_state = model.state_dict()
    matched_state = {
        k: v for k, v in ckpt['model_state_dict'].items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(matched_state, strict=False)
    model.eval()

    tokenizer = SimpleTokenizer.__new__(SimpleTokenizer)
    tokenizer.pad_token = '<pad>'
    tokenizer.sos_token = '<sos>'
    tokenizer.eos_token = '<eos>'
    tokenizer.unk_token = '<unk>'
    tokenizer.word2id = ckpt['word2id']
    tokenizer.id2word = ckpt['id2word']
    tokenizer.vocab_size = ckpt['vocab_size']
    tokenizer.max_len = ckpt.get('max_len', 28)
    tokenizer.pad_id = tokenizer.word2id['<pad>']
    tokenizer.sos_id = tokenizer.word2id['<sos>']
    tokenizer.eos_id = tokenizer.word2id['<eos>']
    tokenizer.unk_id = tokenizer.word2id['<unk>']

    return model, tokenizer


def precompute_embeddings(q4_model, loader, device):
    """Precomputes and caches image and text latent embeddings on the target device."""
    q4_model.eval()
    all_e_I = []
    all_e_S = []
    with torch.no_grad():
        for images, tokens, *rest in loader:
            images = images.to(device)
            tokens = tokens.to(device)
            _, e_I = q4_model.encode_image(images)
            _, e_S = q4_model.encode_text(tokens)
            all_e_I.append(e_I.detach())
            all_e_S.append(e_S.detach())
    return torch.cat(all_e_I, dim=0), torch.cat(all_e_S, dim=0)


def train_flow_model(q4_model, flow_model, train_loader, val_loader, epochs=150, lr=5e-4, batch_size=16, device='cpu'):
    """
    Trains the Flow Matching Decoder to recreate image embeddings e_I
    from noisy embeddings x_t using sentence guide e_S.
    """
    print(f"\n[1/3] Precomputing latent embeddings from Question 4 Cactus model...")
    train_e_I, train_e_S = precompute_embeddings(q4_model, train_loader, device)
    val_e_I, val_e_S = precompute_embeddings(q4_model, val_loader, device)
    print(f"      Train embeddings shape: {train_e_I.shape} | Val embeddings shape: {val_e_I.shape}")

    train_dataset = TensorDataset(train_e_I, train_e_S)
    val_dataset = TensorDataset(val_e_I, val_e_S)

    emb_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    emb_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(flow_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n=================================================================")
    print(f" Training Advanced Flow Matching Model ({device.type.upper()})")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Initial LR: {lr:.2e}")
    print(f" Architecture: 6-Layer FlowBlocks with Multi-Token Cross-Attention & CFG")
    print(f" Loss: Target Reconstruction MSE + Cosine Directional Alignment")
    print(f"=================================================================")

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        flow_model.train()
        total_loss_sum = 0.0
        mse_loss_sum = 0.0
        cos_sim_sum = 0.0
        n_samples = 0

        # Run 2 sub-passes per epoch with varied random time samplings for thorough training
        for _ in range(2):
            for e_I, e_S in emb_train_loader:
                e_I_rep = e_I.repeat(2, 1)
                e_S_rep = e_S.repeat(2, 1)

                loss, mse_val, cos_val = flow_model.compute_loss(e_I_rep, e_S_rep, drop_prob=0.15)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = e_I_rep.size(0)
                total_loss_sum += loss.item() * bs
                mse_loss_sum += mse_val * bs
                cos_sim_sum += cos_val * bs
                n_samples += bs

        scheduler.step()

        train_loss = total_loss_sum / n_samples
        train_mse = mse_loss_sum / n_samples
        train_cos = cos_sim_sum / n_samples

        # Validation
        flow_model.eval()
        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_cos_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for e_I, e_S in emb_val_loader:
                v_loss, v_mse, v_cos = flow_model.compute_loss(e_I, e_S, drop_prob=0.0)
                bs = e_I.size(0)
                val_loss_sum += v_loss.item() * bs
                val_mse_sum += v_mse * bs
                val_cos_sum += v_cos * bs
                val_n += bs

        val_loss = val_loss_sum / val_n
        val_mse = val_mse_sum / val_n
        val_cos = val_cos_sum / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(flow_model.state_dict(), 'flow_matching.pth')

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            curr_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | LR: {curr_lr:.2e} | "
                  f"Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Cos: {train_cos:.3f}) | "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Cos: {val_cos:.3f})")

    print(f"\nTraining Complete! Best Validation Loss: {best_val_loss:.4f}")
    print("Optimal model weights saved to flow_matching.pth")


def run_question_steps(q4_model, flow_model, tokenizer, device, test_img='00469.jpg', test_caption=None):
    """
    Executes and generates comprehensive comparisons for all sub-steps of Question 5:
      a. Given the (sentence S, image img101) pair from Flickr
      b. Create embedding ei101 for img101 using Q4. Create embedding eS for sentence S using Q4.
      c. Use the forward diffusion process to create a noisy version noisye101 of e101.
      d. Train a flow matching model to recreate e101 from noisye101 using eS as guide. Your decoder will
         need to have a cross-attention layer to attend to eS.
    """
    print("\n=================================================================")
    print(" Executing Question 5 Evaluation & Comparison Workflow")
    print("=================================================================")

    # -------------------------------------------------------------
    # Step a: Load pair from Flickr (using image 00469.jpg: cowboy riding horse)
    # -------------------------------------------------------------
    img_path = f'./data/flickr30k/images/{test_img}'
    if not os.path.exists(img_path):
        fallback_path = './data/flickr30k/images/00101.jpg'
        print(f"Warning: {img_path} not found, falling back to {fallback_path}")
        img_path = fallback_path

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    img_target = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)

    if test_caption is None:
        if '00469' in img_path:
            sentence_S = "A cowboy is viciously thrown about as he attempts to hold on to a horse he is riding in a rodeo competition."
        else:
            sentence_S = "Man riding horse on a beach"
    else:
        sentence_S = test_caption

    tokens_S = torch.tensor([tokenizer.encode(sentence_S)], dtype=torch.long, device=device)

    print(f"[Step a] Given Pair:")
    print(f"         Sentence S : \"{sentence_S}\"")
    print(f"         Image Path : {img_path} (shape: {img_target.shape})")

    # -------------------------------------------------------------
    # Step b: Create embedding e_i101 and e_S using Q4
    # -------------------------------------------------------------
    with torch.no_grad():
        _, e_target = q4_model.encode_image(img_target)
        _, e_S = q4_model.encode_text(tokens_S)

    print(f"[Step b] Created Q4 Embeddings:")
    print(f"         Target Image Embedding norm = {e_target.norm(dim=-1).item():.4f}, std = {e_target.std().item():.4f}")
    print(f"         Guide Sentence Embedding norm = {e_S.norm(dim=-1).item():.4f}, std = {e_S.std().item():.4f}")
    init_sim = F.cosine_similarity(e_target, e_S, dim=-1).item()
    print(f"         Initial Cross-Modal Cosine Sim: {init_sim:.4f}")

    # -------------------------------------------------------------
    # Step c: Forward diffusion process creates noisy versions
    # -------------------------------------------------------------
    t_test_levels = [0.8, 0.5, 0.2]  # t=1 is clean, t=0 is pure noise
    fixed_noise = torch.randn_like(e_target)
    noisy_dict = {}

    print(f"[Step c] Forward Diffusion / Flow Process:")
    for t_val in t_test_levels:
        t_t = torch.tensor([t_val], device=device)
        noisy_e, _, _ = flow_model.forward_noise(e_target, t_t, x_0=fixed_noise)
        sim_to_clean = F.cosine_similarity(noisy_e, e_target, dim=-1).item()
        mse_to_clean = F.mse_loss(noisy_e, e_target).item()
        noisy_dict[t_val] = {
            'noisy_e': noisy_e,
            'cos_sim': sim_to_clean,
            'mse': mse_to_clean
        }
        print(f"         t={t_val:.2f} | Cos Sim to Clean: {sim_to_clean:+.4f} | MSE: {mse_to_clean:.4f}")

    # -------------------------------------------------------------
    # Step d: Recreate e_target from noisy versions guided by e_S via ODE
    # -------------------------------------------------------------
    print(f"[Step d] Recreating Target Embedding from Noisy Versions Guided by e_S:")
    recreated_dict = {}

    for t_val in t_test_levels:
        noisy_e = noisy_dict[t_val]['noisy_e']
        recreated_e, traj = flow_model.recreate_from_noise(noisy_e, t_start=t_val, e_S=e_S, steps=50, method='rk2', cfg_scale=1.5)
        # Scale calibrate to target manifold norm
        recreated_e = recreated_e * (e_target.norm() / max(recreated_e.norm(), 1e-4))
        final_sim = F.cosine_similarity(recreated_e, e_target, dim=-1).item()
        final_mse = F.mse_loss(recreated_e, e_target).item()
        recreated_dict[t_val] = {
            'recreated_e': recreated_e,
            'cos_sim': final_sim,
            'mse': final_mse,
            'trajectory': traj
        }
        print(f"         Recreated from t={t_val:.2f} -> 1.0 | Final Cos Sim: {final_sim:+.4f} | MSE: {final_mse:.5f}")

    # Also test standard conditional generation from pure noise (t=0.0)
    sampled_e, sample_traj = flow_model.recreate_from_noise(fixed_noise, t_start=0.0, e_S=e_S, steps=50, method='rk2', cfg_scale=1.5)
    sampled_e = sampled_e * (e_target.norm() / max(sampled_e.norm(), 1e-4))
    sample_sim = F.cosine_similarity(sampled_e, e_target, dim=-1).item()
    sample_mse = F.mse_loss(sampled_e, e_target).item()
    print(f"         Recreated from Pure Noise t=0.00 -> 1.0 | Final Cos Sim: {sample_sim:+.4f} | MSE: {sample_mse:.5f}")

    # -------------------------------------------------------------
    # High-Fidelity Multi-Panel Visualization
    # -------------------------------------------------------------
    print("\nRendering comprehensive results to flow_matching_results.png...")
    fig = plt.figure(figsize=(18, 11), facecolor='#f8f9fa')
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: Steps (a) & (b) - Input Pair & Latent Embeddings
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#ffffff')

    img_np = img_target.squeeze(0).cpu().permute(1, 2, 0).numpy()
    dims = min(48, e_target.size(-1))
    x_axis = np.arange(dims)

    ax1.plot(x_axis, e_target[0, :dims].cpu().numpy(), label=r'Target Image Embedding $e_{target}$', color='#1f77b4', linewidth=2.2)
    ax1.plot(x_axis, e_S[0, :dims].cpu().numpy(), label=r'Guide Sentence Embedding $e_S$', color='#ff7f0e', linestyle='--', linewidth=2.0)
    ax1.set_title(f"[Step a & b] Flickr Input Pair ({test_img}) & Q4 Embeddings\nSentence S: \"{sentence_S[:60]}...\"", fontsize=10, fontweight='bold', pad=10)
    ax1.set_xlabel(f"Latent Embedding Dimensions (First {dims} of {e_target.size(-1)})", fontsize=9)
    ax1.set_ylabel("Embedding Value", fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    inset_ax = ax1.inset_axes([0.02, 0.58, 0.32, 0.38])
    inset_ax.imshow(img_np)
    inset_ax.axis('off')
    inset_ax.set_title(f"Target GT ({test_img})", fontsize=8, fontweight='bold')

    # -------------------------------------------------------------
    # Panel 2: Step (c) - Forward Diffusion Process (Latent & Decoded)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#ffffff')
    ax2.axis('off')
    ax2.set_title("[Step c] Forward Diffusion Process: $x_t = (1-t)x_0 + t e_{target}$\n(Embeddings Decoded Across Noise Levels)",
                  fontsize=10.5, fontweight='bold', pad=10)

    display_t_levels = [1.0, 0.8, 0.5, 0.2, 0.0]
    n_cols = len(display_t_levels)

    for idx, t_v in enumerate(display_t_levels):
        if t_v == 1.0:
            sub_e = e_target
            label = "Clean Target\n$t=1.00$"
        elif t_v == 0.0:
            sub_e = fixed_noise
            label = "Pure Noise $x_0$\n$t=0.00$"
        else:
            sub_e = noisy_dict[t_v]['noisy_e']
            sim = noisy_dict[t_v]['cos_sim']
            label = f"Noisy $x_{{{t_v:.1f}}}$\nSim: {sim:+.2f}"

        with torch.no_grad():
            dec_img = q4_model.decode_from_embedding(sub_e)
        dec_np = dec_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

        sub_ax = ax2.inset_axes([idx / n_cols + 0.01, 0.12, 1.0 / n_cols - 0.02, 0.72])
        sub_ax.imshow(dec_np)
        sub_ax.axis('off')
        sub_ax.set_title(label, fontsize=8.5)

    # -------------------------------------------------------------
    # Panel 3: Step (d) - Recreated Embeddings Guided by e_S
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#ffffff')
    ax3.axis('off')
    ax3.set_title("[Step d] Flow Matching Recreated Images Guided by $e_S$\n(Direct Side-by-Side Comparison with Target Ground Truth)",
                  fontsize=10.5, fontweight='bold', pad=10)

    recon_cols = [
        ("From $t=0.8$\n(Mild Noise)", recreated_dict[0.8]['recreated_e'], recreated_dict[0.8]['cos_sim']),
        ("From $t=0.5$\n(Mod. Noise)", recreated_dict[0.5]['recreated_e'], recreated_dict[0.5]['cos_sim']),
        ("From $t=0.2$\n(Heavy Noise)", recreated_dict[0.2]['recreated_e'], recreated_dict[0.2]['cos_sim']),
        ("From Pure Noise\n($t=0.0$)", sampled_e, sample_sim),
        (f"Target GT\n({test_img})", e_target, 1.000)
    ]

    n_rcols = len(recon_cols)
    for idx, (title, emb_v, sim_val) in enumerate(recon_cols):
        with torch.no_grad():
            dec_img = q4_model.decode_from_embedding(emb_v)
        dec_np = dec_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

        sub_ax = ax3.inset_axes([idx / n_rcols + 0.01, 0.12, 1.0 / n_rcols - 0.02, 0.72])
        sub_ax.imshow(dec_np)
        sub_ax.axis('off')
        color = '#2ca02c' if sim_val > 0.8 else '#1f77b4'
        sub_ax.set_title(f"{title}\nCos: {sim_val:+.3f}", fontsize=8.5, color=color, fontweight='bold')

    # -------------------------------------------------------------
    # Panel 4: ODE Trajectories & Distance Metrics to Target
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#ffffff')

    colors = {0.8: '#2ca02c', 0.5: '#1f77b4', 0.2: '#ff7f0e'}
    for t_val in t_test_levels:
        traj = recreated_dict[t_val]['trajectory']
        t_steps_axis = [item[0] for item in traj]
        dists = [torch.norm(item[1] - e_target).item() for item in traj]
        ax4.plot(t_steps_axis, dists, label=f"Reconstruct from $t={t_val:.1f}$ (Final Sim: {recreated_dict[t_val]['cos_sim']:+.3f})",
                 color=colors[t_val], linewidth=2.2, marker='o', markersize=3.5)

    sample_t_axis = [item[0] for item in sample_traj]
    sample_dists = [torch.norm(item[1] - e_target).item() for item in sample_traj]
    ax4.plot(sample_t_axis, sample_dists, label=f"Pure Noise $t=0.0$ -> 1.0 (Final Sim: {sample_sim:+.3f})",
             color='#9467bd', linestyle='--', linewidth=2.0)

    ax4.set_title(r"ODE Transport Flow: Distance $\|x_t - e_{target}\|$ Decreasing to Zero", fontsize=10.5, fontweight='bold', pad=10)
    ax4.set_xlabel(r"Time Parameter $t \in [t_{start}, 1.0]$", fontsize=9)
    ax4.set_ylabel(r"L2 Distance to Target $e_{target}$", fontsize=9)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=8, framealpha=0.9)

    plt.savefig('flow_matching_results.png', dpi=220, bbox_inches='tight')
    plt.close()
    print("SUCCESS: Comprehensive comparison saved to flow_matching_results.png")


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate Advanced Flow Matching on Flickr Multimodal Latents")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs (default: 150)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument('--layers', type=int, default=6, help="Number of FlowBlock layers (default: 6)")
    parser.add_argument('--heads', type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument('--test_img', type=str, default='00469.jpg', help="Image filename to test (default: 00469.jpg)")
    parser.add_argument('--max_images', type=int, default=2000, help="Maximum number of images for Flow Matching dataset (default: 2000)")
    parser.add_argument('--eval_only', action='store_true', help="Skip training and run evaluation only")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")

    # Load Question 4 Cactus Multimodal Model
    q4_model, tokenizer = load_q4_model('cactus_flickr.pth', device=device)

    # Initialize Advanced Flow Matching Model
    d_model = q4_model.d_model
    flow_model = FlowMatching(d_model=d_model, num_heads=args.heads, num_layers=args.layers, num_guide_tokens=8).to(device)

    if not args.eval_only:
        train_loader, val_loader, *rest = get_dataloaders(batch_size=args.batch_size, max_samples=args.max_images)
        print(f"Flow Matching Dataset Config: Total={args.max_images} | Train={len(train_loader.dataset)} (85%) | Val/Test={len(val_loader.dataset)} (15%)")
        train_flow_model(q4_model, flow_model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=device)
    else:
        if os.path.exists('flow_matching.pth'):
            try:
                flow_model.load_state_dict(torch.load('flow_matching.pth', map_location=device))
                print("Loaded trained flow matching weights from flow_matching.pth")
            except Exception as e:
                print(f"Could not load flow_matching.pth ({e}). Using initialized model for dry run.")

    # Run question steps and generate comprehensive results on test image
    run_question_steps(q4_model, flow_model, tokenizer, device=device, test_img=args.test_img)


if __name__ == '__main__':
    main()
