"""
Training Script for Adaptive Diffusion Fusion Model
Trains the model on M3FD dataset (200 training pairs).
Run from command line: python run_train.py
"""
import os
import sys
import copy
import argparse
import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from models.adaptive_fusion_net import AdaptiveFusionModel
from models.adaptive_diffusion import GaussianDiffusion
from data.m3fd_dataset import M3FDTrainDataset


def update_ema(ema_model, model, decay=0.9999):
    """Update exponential moving average model."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def save_checkpoint(model, ema_model, optimizer, scheduler, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_state_dict': ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, path)


@torch.no_grad()
def generate_samples(model, diffusion, dataset, device, output_dir, num_samples=4):
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


def run_training(
    data_dir=None,
    output_dir=None,
    checkpoint_dir=None,
    device="cuda",
    epochs=100,
    batch_size=4,
    lr=1e-4,
    seed=42,
    resume_from=None,
    save_every=10,
    sample_every=10,
    ema_decay=0.9999,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            loss = diffusion.p_losses(model, gt_fused, t, ir, vis)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

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


def main():
    parser = argparse.ArgumentParser(
        description="Train Adaptive Diffusion Fusion Model on M3FD"
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=10)

    args = parser.parse_args()

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
