"""
Training Script for Adaptive Diffusion Fusion Model
Trains the model on M3FD dataset (200 training pairs).

Supports two modes:
  - Standard training: python run_train.py
  - Adaptive training: python run_train.py --adaptive --phase 1 --resume_v1 checkpoints/best.pt
                        python run_train.py --adaptive --phase 2 --resume checkpoints/adaptive_best.pt
"""
import os
import sys
import copy
import argparse
import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.adaptive_fusion_net import AdaptiveFusionModel, LatentAdaptiveFusionModel
from models.adaptive_diffusion import GaussianDiffusion
from models.content_analyzer import (
    ContentAnalyzer,
    AdaptiveTimestepSelector,
    ContentAdaptiveNoiseSchedule,
)
from models.feature_memory import FeatureMemoryBank, DualModalityFeatureMemory
from models.vae import FusionVAE
from data.m3fd_dataset import M3FDTrainDataset


def update_ema(ema_model, model, decay=0.9999):
    """Update exponential moving average model."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def save_checkpoint(model, ema_model, optimizer, scheduler, epoch, loss, path,
                    adaptive_modules=None):
    """Save training checkpoint."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    if adaptive_modules is not None:
        ckpt['version'] = 2
        ckpt['content_analyzer_state_dict'] = \
            adaptive_modules['content_analyzer'].state_dict()
        ckpt['timestep_selector_state_dict'] = \
            adaptive_modules['timestep_selector'].state_dict()
        ckpt['noise_schedule_state_dict'] = \
            adaptive_modules['noise_schedule'].state_dict()
        ckpt['feature_memory_state_dict'] = \
            adaptive_modules['feature_memory'].state_dict()

    torch.save(ckpt, path)


@torch.no_grad()
def generate_samples(model, diffusion, dataset, device, output_dir,
                     num_samples=4):
    """Generate sample fused images for visual inspection."""
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from torchvision.utils import save_image

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        ir = sample['ir'].unsqueeze(0).to(device)
        vis = sample['vis'].unsqueeze(0).to(device)

        fused = diffusion.ddim_sample_loop(
            model, (1, 3, 256, 256), ir, vis,
            ddim_steps=50, verbose=False
        )

        fused_01 = (fused.clamp(-1, 1) + 1) / 2
        ir_01 = (ir + 1) / 2
        vis_01 = (vis + 1) / 2

        grid = torch.cat([ir_01, vis_01, fused_01], dim=3)
        save_image(grid, output_path / f"sample_{sample['name']}.png")

    model.train()


@torch.no_grad()
def compute_oracle_complexity(model, diffusion, ir, vis, gt_fused, device):
    """
    Self-supervised oracle complexity: measure how hard each image is
    to denoise at mid-range timesteps. Higher reconstruction error = more complex.

    Returns:
        complexity_targets: [B, 1] in [0, 1]
    """
    B = ir.shape[0]
    test_timesteps = [200, 400, 600]
    total_error = torch.zeros(B, device=device)

    for t_val in test_timesteps:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(gt_fused)
        x_noisy = diffusion.q_sample(gt_fused, t, noise=noise)
        noise_pred = model(x_noisy, t, ir, vis)
        per_sample_mse = F.mse_loss(
            noise_pred, noise, reduction='none'
        ).mean(dim=[1, 2, 3])
        total_error += per_sample_mse

    avg_error = total_error / len(test_timesteps)
    complexity_targets = torch.sigmoid((avg_error - 0.05) * 20)
    return complexity_targets.unsqueeze(1)


# ================================================================
# Standard Training (unchanged from original)
# ================================================================

def run_training(
    data_dir=None,
    output_dir=None,
    checkpoint_dir=None,
    device="cuda",
    epochs=500,
    batch_size=4,
    lr=2e-4,
    seed=42,
    resume_from=None,
    save_every=25,
    sample_every=50,
    ema_decay=0.9999,
    warmup_epochs=10,
):
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd", "train")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "training")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 60)
    print("ADAPTIVE DIFFUSION FUSION - TRAINING")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Data:       {data_dir}")
    print("=" * 60)

    try:
        dataset = M3FDTrainDataset(
            root_dir=data_dir,
            image_size=(256, 256),
            augment=True,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("Run setup first: python setup_m3fd.py --source_dir /path/to/M3FD")
        sys.exit(1)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda")
    )

    model = AdaptiveFusionModel().to(device)
    ema_model = copy.deepcopy(model)
    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,} ({total_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        ema_model.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}")

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print(f"  AMP:        {'enabled' if use_amp else 'disabled'}")
    print(f"\nStarting training...")
    best_loss = float('inf')
    loss_history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            gt_fused = batch['gt_fused'].to(device)

            B = ir.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = diffusion.p_losses(model, gt_fused, t, ir, vis)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, decay=ema_decay)

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, avg_loss,
                os.path.join(checkpoint_dir, "best.pt")
            )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, avg_loss,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            )

        if (epoch + 1) % sample_every == 0:
            print("  Generating samples...")
            sample_dataset = M3FDTrainDataset(
                root_dir=data_dir, image_size=(256, 256), augment=False
            )
            generate_samples(
                ema_model, diffusion, sample_dataset, device,
                os.path.join(output_dir, f"samples_epoch_{epoch+1}")
            )

    save_checkpoint(
        model, ema_model, optimizer, scheduler, epochs - 1, avg_loss,
        os.path.join(checkpoint_dir, "final.pt")
    )

    log = {
        'phase': 'training',
        'timestamp': datetime.now().isoformat(),
        'dataset': 'M3FD',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'best_loss': best_loss,
        'final_loss': avg_loss,
        'loss_history': loss_history,
        'model_params': total_params,
        'device': device,
    }
    with open(os.path.join(output_dir, "train_log.json"), 'w') as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best loss:    {best_loss:.6f}")
    print(f"  Final loss:   {avg_loss:.6f}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    print(f"  Best model:   {os.path.join(checkpoint_dir, 'best.pt')}")
    print(f"\nNext step: python run_test.py")
    print("=" * 60)

    return loss_history


# ================================================================
# Adaptive Diffusion Training
# ================================================================

def run_adaptive_training(
    data_dir=None,
    output_dir=None,
    checkpoint_dir=None,
    device="cuda",
    phase=1,
    epochs=50,
    batch_size=4,
    lr=1e-4,
    seed=42,
    resume_from=None,
    resume_v1=None,
    save_every=10,
    sample_every=25,
    ema_decay=0.9999,
    warmup_epochs=5,
):
    """
    Adaptive diffusion training with two phases:

    Phase 1: Freeze backbone, train adaptive modules (ContentAnalyzer,
             NoiseSchedule, TimestepSelector, FeatureMemory projections).
             Uses self-supervised oracle complexity as target.

    Phase 2: Unfreeze all, joint fine-tuning with content-adaptive
             noise schedule and schedule regularization.
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd", "train")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "adaptive_training")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 60)
    print(f"ADAPTIVE DIFFUSION FUSION - PHASE {phase} TRAINING")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Phase:      {phase}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")
    print(f"  Data:       {data_dir}")
    print("=" * 60)

    try:
        dataset = M3FDTrainDataset(
            root_dir=data_dir, image_size=(256, 256), augment=True
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\nError: {e}")
        print("Run setup first: python setup_m3fd.py --source_dir /path/to/M3FD")
        sys.exit(1)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda")
    )

    # --- Initialize models ---
    model = AdaptiveFusionModel().to(device)
    diffusion = GaussianDiffusion(num_timesteps=1000, beta_schedule='linear')

    # Adaptive modules
    content_analyzer = ContentAnalyzer(descriptor_dim=64).to(device)
    timestep_selector = AdaptiveTimestepSelector(descriptor_dim=64).to(device)
    noise_schedule = ContentAdaptiveNoiseSchedule(descriptor_dim=64).to(device)
    feature_memory = FeatureMemoryBank().to(device)

    adaptive_modules = {
        'content_analyzer': content_analyzer,
        'timestep_selector': timestep_selector,
        'noise_schedule': noise_schedule,
        'feature_memory': feature_memory,
    }

    # --- Load checkpoint ---
    start_epoch = 0

    if resume_v1 and os.path.exists(resume_v1):
        print(f"\nLoading v1 backbone from: {resume_v1}")
        ckpt = torch.load(resume_v1, map_location=device, weights_only=False)
        state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict'))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (expected for new modules): {len(missing)}")
        print("  Backbone loaded successfully")

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming adaptive training from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'content_analyzer_state_dict' in ckpt:
            content_analyzer.load_state_dict(
                ckpt['content_analyzer_state_dict'])
            timestep_selector.load_state_dict(
                ckpt['timestep_selector_state_dict'])
            noise_schedule.load_state_dict(
                ckpt['noise_schedule_state_dict'])
            feature_memory.load_state_dict(
                ckpt['feature_memory_state_dict'])
            print("  Adaptive modules loaded")
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resumed at epoch {start_epoch}")

    ema_model = copy.deepcopy(model)

    # --- Optimizer setup (differs by phase) ---
    if phase == 1:
        # Freeze backbone, train only adaptive modules
        for param in model.parameters():
            param.requires_grad = False
        trainable_params = (
            list(content_analyzer.parameters()) +
            list(timestep_selector.parameters()) +
            list(noise_schedule.parameters()) +
            list(feature_memory.parameters())
        )
        print(f"  Phase 1: Backbone FROZEN, training adaptive modules only")
    else:
        # Unfreeze all, different LR groups
        for param in model.parameters():
            param.requires_grad = True
        trainable_params = [
            {'params': model.parameters(), 'lr': lr * 0.1},
            {'params': content_analyzer.parameters()},
            {'params': timestep_selector.parameters()},
            {'params': noise_schedule.parameters()},
            {'params': feature_memory.parameters()},
        ]
        print(f"  Phase 2: Joint fine-tuning (backbone LR={lr*0.1:.1e})")

    adaptive_param_count = sum(
        p.numel() for m in adaptive_modules.values() for p in m.parameters()
    )
    backbone_params = sum(p.numel() for p in model.parameters())
    print(f"  Backbone params:  {backbone_params:,} ({backbone_params/1e6:.1f}M)")
    print(f"  Adaptive params:  {adaptive_param_count:,} ({adaptive_param_count/1e6:.1f}M)")
    total_all = backbone_params + adaptive_param_count
    print(f"  Total params:     {total_all:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    base_alphas_cumprod = diffusion.alphas_cumprod

    print(f"  AMP:        {'enabled' if use_amp else 'disabled'}")
    print(f"\nStarting adaptive training phase {phase}...")
    best_loss = float('inf')
    loss_history = []

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        content_analyzer.train()
        timestep_selector.train()
        noise_schedule.train()
        feature_memory.train()

        epoch_loss = 0.0
        epoch_denoise = 0.0
        epoch_complexity = 0.0
        num_batches = 0

        pbar = tqdm(dataloader,
                    desc=f"Phase {phase} Epoch {epoch+1}/{start_epoch+epochs}")
        for batch in pbar:
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            gt_fused = batch['gt_fused'].to(device)

            B = ir.shape[0]
            t = torch.randint(0, 1000, (B,), device=device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                # --- Content analysis ---
                content_desc, complexity_map, complexity_scalar = \
                    content_analyzer(ir, vis)

                # --- Adaptive noise schedule ---
                adaptive_betas, adaptive_alphas_cumprod = \
                    noise_schedule(content_desc)

                if phase == 1:
                    # Phase 1: Standard denoising + complexity supervision
                    denoise_loss = diffusion.p_losses(
                        model, gt_fused, t, ir, vis
                    )

                    # Oracle complexity target
                    oracle_complexity = compute_oracle_complexity(
                        model, diffusion, ir, vis, gt_fused, device
                    )
                    complexity_loss = F.mse_loss(
                        complexity_scalar, oracle_complexity
                    )

                    # Diversity: prevent constant predictions
                    diversity_loss = torch.clamp(
                        0.05 - complexity_scalar.var(), min=0
                    )

                    loss = denoise_loss + complexity_loss + 0.1 * diversity_loss

                else:
                    # Phase 2: Adaptive denoising with content-dependent schedule
                    loss, loss_dict = diffusion.adaptive_p_losses(
                        model, gt_fused, t, ir, vis,
                        adaptive_alphas_cumprod,
                        base_alphas_cumprod=base_alphas_cumprod,
                        schedule_reg_weight=0.01,
                    )

                    diversity_loss = torch.clamp(
                        0.05 - complexity_scalar.var(), min=0
                    )
                    loss = loss + 0.1 * diversity_loss

                    denoise_loss = torch.tensor(loss_dict['denoising'])
                    complexity_loss = torch.tensor(loss_dict['schedule_reg'])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            all_params = []
            for group in optimizer.param_groups:
                if isinstance(group['params'], list):
                    all_params.extend(group['params'])
                else:
                    all_params.append(group['params'])
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            scaler.step(optimizer)
            scaler.update()

            if phase == 2:
                update_ema(ema_model, model, decay=ema_decay)

            epoch_loss += loss.item()
            epoch_denoise += denoise_loss.item()
            epoch_complexity += complexity_loss.item()
            num_batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cmplx=f"{complexity_scalar.mean().item():.3f}"
            )

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_denoise = epoch_denoise / max(num_batches, 1)
        avg_complexity = epoch_complexity / max(num_batches, 1)
        loss_history.append(avg_loss)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}, "
              f"denoise={avg_denoise:.6f}, "
              f"complexity={avg_complexity:.6f}, lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, avg_loss,
                os.path.join(checkpoint_dir, "adaptive_best.pt"),
                adaptive_modules=adaptive_modules
            )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, ema_model, optimizer, scheduler, epoch, avg_loss,
                os.path.join(checkpoint_dir,
                             f"adaptive_checkpoint_epoch_{epoch+1}.pt"),
                adaptive_modules=adaptive_modules
            )

        if (epoch + 1) % sample_every == 0:
            print("  Generating samples...")
            sample_dataset = M3FDTrainDataset(
                root_dir=data_dir, image_size=(256, 256), augment=False
            )
            generate_samples(
                ema_model if phase == 2 else model,
                diffusion, sample_dataset, device,
                os.path.join(output_dir, f"samples_epoch_{epoch+1}")
            )

    # Save final
    save_checkpoint(
        model, ema_model, optimizer, scheduler,
        start_epoch + epochs - 1, avg_loss,
        os.path.join(checkpoint_dir, f"adaptive_phase{phase}_final.pt"),
        adaptive_modules=adaptive_modules
    )

    log = {
        'phase': f'adaptive_phase_{phase}',
        'timestamp': datetime.now().isoformat(),
        'dataset': 'M3FD',
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'best_loss': best_loss,
        'final_loss': avg_loss,
        'loss_history': loss_history,
        'backbone_params': backbone_params,
        'adaptive_params': adaptive_param_count,
        'device': device,
    }
    with open(os.path.join(output_dir,
                           f"adaptive_phase{phase}_log.json"), 'w') as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print(f"ADAPTIVE PHASE {phase} TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best loss:    {best_loss:.6f}")
    print(f"  Final loss:   {avg_loss:.6f}")
    print(f"  Checkpoints:  {checkpoint_dir}")
    if phase == 1:
        ckpt_path = os.path.join(checkpoint_dir, 'adaptive_best.pt')
        print(f"\nNext step: python run_train.py --adaptive --phase 2 "
              f"--resume {ckpt_path}")
    else:
        print(f"\nNext step: python run_test.py --adaptive")
    print("=" * 60)

    return loss_history


# ================================================================
# VAE Pretraining (Phase 0 for latent-space mode)
# ================================================================

def run_vae_pretraining(
    data_dir=None, output_dir=None, checkpoint_dir=None,
    device="cuda", epochs=100, batch_size=4, lr=1e-4,
    seed=42, save_every=10,
):
    """
    Pretrain the FusionVAE to encode/decode IR and VIS images.
    Must be run before latent-space adaptive training.

    Usage: python run_train.py --vae --epochs 100
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "m3fd", "train")
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "output", "vae_training")
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 60)
    print("FUSION VAE - PRETRAINING")
    print("=" * 60)

    dataset = M3FDTrainDataset(root_dir=data_dir, image_size=(256, 256), augment=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device == "cuda"))

    vae = FusionVAE(latent_channels=4).to(device)
    vae_params = sum(p.numel() for p in vae.parameters())
    print(f"  VAE params: {vae_params:,} ({vae_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_loss = float('inf')

    for epoch in range(epochs):
        vae.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = vae(ir, vis)

                # IR reconstruction + KL
                ir_loss, ir_recon, ir_kl = FusionVAE.vae_loss(
                    out['recon_ir'], ir, out['mean_ir'], out['logvar_ir']
                )
                # VIS reconstruction + KL
                vis_loss, vis_recon, vis_kl = FusionVAE.vae_loss(
                    out['recon_vis'], vis, out['mean_vis'], out['logvar_vis']
                )
                loss = ir_loss + vis_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, "vae_best.pt"))

        if (epoch + 1) % save_every == 0:
            torch.save({
                'vae_state_dict': vae.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, f"vae_epoch_{epoch+1}.pt"))

    print(f"\nVAE pretraining complete! Best loss: {best_loss:.6f}")
    print(f"Next: python run_train.py --latent --phase 1 --epochs 50")


def main():
    parser = argparse.ArgumentParser(
        description="Train Adaptive Diffusion Fusion Model on M3FD"
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=10)

    # Mode selection
    parser.add_argument("--adaptive", action="store_true",
                        help="Adaptive diffusion training (pixel-space)")
    parser.add_argument("--vae", action="store_true",
                        help="Pretrain the FusionVAE (phase 0)")
    parser.add_argument("--latent", action="store_true",
                        help="Latent-space adaptive training (requires pretrained VAE)")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Training phase (1=freeze backbone, 2=joint fine-tune)")
    parser.add_argument("--resume_v1", type=str, default=None,
                        help="Path to v1 checkpoint for backbone init")

    args = parser.parse_args()

    if args.vae:
        run_vae_pretraining(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            save_every=args.save_every,
        )
    elif args.adaptive or args.latent:
        run_adaptive_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            phase=args.phase,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            resume_from=args.resume,
            resume_v1=args.resume_v1,
            save_every=args.save_every,
            sample_every=args.sample_every,
        )
    else:
        run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            resume_from=args.resume,
            save_every=args.save_every,
            sample_every=args.sample_every,
        )


if __name__ == "__main__":
    main()
