"""
train.py — Full 2-phase GAN training for YD-Upscale anime super-resolution.

Phase 1  (warmup, first N iterations):
    Train generator only with L1 + VGG perceptual loss.
    All weights initialised from scratch (default PyTorch init).

Phase 2  (GAN, remaining iterations):
    Activate the U-Net discriminator with spectral normalisation.
    Full adversarial training: L1 + perceptual + vanilla GAN loss.
    Alternating generator/discriminator optimiser steps.

Key settings
------------
    Generator   : RRDBNet nb=6, nf=64, gc=32, scale=4
    Discriminator: UNetDiscriminatorSN nf=64
    G optimizer : Adam lr=1e-4, betas=(0.9, 0.99)
    D optimizer : Adam lr=1e-4, betas=(0.9, 0.99)
    AMP         : mixed-precision on CUDA
    EMA         : decay=0.999, saved as params_ema
    Batch       : 16  (256×256 HR → 64×64 LR)
    Grad clip   : max_norm=1.0

Checkpoints → checkpoints/
    YD_UPSCALE_anime_x4_iter_N.pth
    YD_UPSCALE_anime_x4_best.pth

Usage
-----
    python train.py
    python train.py --total-iter 400000 --warmup-iter 50000 --batch-size 16
    python train.py --resume checkpoints/YD_UPSCALE_anime_x4_iter_100000.pth
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from yd_upscale.data.dataloader_factory import build_train_val_dataloaders
from yd_upscale.losses.gan_loss import DiscriminatorLoss, GeneratorLoss
from yd_upscale.models.discriminator import UNetDiscriminatorSN
from yd_upscale.models.rrdbnet import RRDBNet


# ============================================================================
# EMA helper
# ============================================================================

class ModelEMA:
    """
    Exponential Moving Average of generator weights.
    shadow = decay * shadow + (1-decay) * current
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow.state_dict()


# ============================================================================
# PSNR (for logging)
# ============================================================================

@torch.no_grad()
def calc_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    PSNR between two (B,3,H,W) tensors in [0,1].
    Returns average over batch.
    """
    mse = (sr - hr).pow(2).mean(dim=[1, 2, 3])        # per-image MSE
    psnr = -10.0 * torch.log10(mse + 1e-8)
    return psnr.mean().item()


# ============================================================================
# Validation
# ============================================================================

def validate(
    model: nn.Module,
    val_loader,
    g_criterion: GeneratorLoss,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Returns (avg_val_loss, avg_psnr) over the validation set."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                sr = model(lr).clamp(0.0, 1.0)
                loss, _ = g_criterion(sr, hr, disc_fake_pred=None)

            total_loss += loss.item()
            total_psnr += calc_psnr(sr.float(), hr.float())
            n += 1

    return total_loss / max(n, 1), total_psnr / max(n, 1)


# ============================================================================
# Preview helper
# ============================================================================

def save_preview(
    model: nn.Module,
    device: torch.device,
    input_path: Path,
    output_path: Path,
    use_amp: bool,
) -> None:
    if not input_path.exists():
        return
    model.eval()
    img = Image.open(input_path).convert("RGB")
    t = TF.to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            sr = model(t).clamp(0.0, 1.0)
    TF.to_pil_image(sr.squeeze(0).float().cpu()).save(output_path)
    print(f"  [Preview] {output_path}")


# ============================================================================
# Resume from checkpoint
# ============================================================================

def resume_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    net_g: nn.Module,
    net_d: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    ema: ModelEMA,
) -> tuple[int, float]:
    """
    Resume training from a saved checkpoint.
    Returns (start_iteration, best_val_loss).
    """
    print(f"Resuming from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    net_g.load_state_dict(ckpt["model_state_dict"])
    if "disc_state_dict" in ckpt and ckpt["disc_state_dict"] is not None:
        net_d.load_state_dict(ckpt["disc_state_dict"])
    opt_g.load_state_dict(ckpt["opt_g_state_dict"])
    if "opt_d_state_dict" in ckpt and ckpt["opt_d_state_dict"] is not None:
        opt_d.load_state_dict(ckpt["opt_d_state_dict"])
    if "params_ema" in ckpt:
        ema.shadow.load_state_dict(ckpt["params_ema"])

    start_iter = ckpt.get("iteration", 0)
    best_val   = ckpt.get("best_val_loss", float("inf"))
    print(f"  Resumed at iteration {start_iter}, best_val_loss={best_val:.6f}")
    return start_iter, best_val


# ============================================================================
# Argument parser
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YD-Upscale anime GAN training")

    # Iterations
    p.add_argument("--total-iter",    type=int,   default=400_000,
                   help="Total training iterations")
    p.add_argument("--warmup-iter",   type=int,   default=50_000,
                   help="Phase 1 (G-only) iterations before enabling discriminator")

    # Data
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--num-workers",   type=int,   default=4)
    p.add_argument("--scale",         type=int,   default=4)
    p.add_argument("--hr-crop",       type=int,   default=256)

    # Optimiser
    p.add_argument("--lr-g",          type=float, default=1e-4,
                   help="Generator learning rate")
    p.add_argument("--lr-d",          type=float, default=1e-4,
                   help="Discriminator learning rate")

    # Loss weights
    p.add_argument("--lambda-l1",     type=float, default=1.0)
    p.add_argument("--lambda-percep", type=float, default=1.0)
    p.add_argument("--lambda-gan",    type=float, default=0.1)

    # EMA
    p.add_argument("--ema-decay",     type=float, default=0.999)

    # Resume
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint to resume from")

    # Logging
    p.add_argument("--log-every",     type=int,   default=100,
                   help="Print losses/PSNR every N iterations")
    p.add_argument("--save-every",    type=int,   default=5000,
                   help="Save checkpoint every N iterations")
    p.add_argument("--val-every",     type=int,   default=5000,
                   help="Run validation every N iterations")

    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()

    # ---- Paths ----
    project_root = Path(__file__).resolve().parent
    manifest_dir = project_root / "data" / "manifests"
    ckpt_dir     = project_root / "checkpoints"
    preview_dir  = ckpt_dir / "epoch_previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_inputs = [
        project_root / "Testing" / "Test.jpg",
        project_root / "Testing" / "Tests.jpg",
        project_root / "Testing" / "testtt.png",
    ]

    # ---- Device / AMP ----
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}  |  Batch: {args.batch_size}")

    # ---- Data ----
    train_loader, val_loader = build_train_val_dataloaders(
        manifest_dir=manifest_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        hr_crop_size=args.hr_crop,
    )
    print(
        f"Train: {len(train_loader.dataset)} imgs  |  "
        f"Val: {len(val_loader.dataset)} imgs  |  "
        f"HR: {args.hr_crop}  LR: {args.hr_crop // args.scale}"
    )

    # ---- Models ----
    # Generator (anime 6B)
    net_g = RRDBNet(
        in_nc=3, out_nc=3, nf=64, nb=6, gc=32, scale=args.scale,
    ).to(device)

    # Discriminator (U-Net with spectral norm)
    net_d = UNetDiscriminatorSN(
        in_channels=3, nf=64, skip_connection=True,
    ).to(device)

    # ---- EMA ----
    ema = ModelEMA(net_g, decay=args.ema_decay)

    # ---- Losses ----
    g_criterion = GeneratorLoss(
        lambda_l1=args.lambda_l1,
        lambda_percep=args.lambda_percep,
        lambda_gan=args.lambda_gan,
    ).to(device)

    d_criterion = DiscriminatorLoss().to(device)

    # ---- Optimisers ----
    opt_g = Adam(net_g.parameters(), lr=args.lr_g, betas=(0.9, 0.99))
    opt_d = Adam(net_d.parameters(), lr=args.lr_d, betas=(0.9, 0.99))

    # ---- Schedulers ----
    # Cosine anneal over the total iterations
    total_epochs_approx = max(args.total_iter // max(len(train_loader), 1), 1)
    sched_g = CosineAnnealingLR(opt_g, T_max=total_epochs_approx, eta_min=1e-7)
    sched_d = CosineAnnealingLR(opt_d, T_max=total_epochs_approx, eta_min=1e-7)

    # ---- AMP scalers ----
    scaler_g = GradScaler("cuda", enabled=use_amp)
    scaler_d = GradScaler("cuda", enabled=use_amp)

    # ---- Resume ----
    start_iter     = 0
    best_val_loss  = float("inf")
    if args.resume:
        start_iter, best_val_loss = resume_from_checkpoint(
            Path(args.resume), device, net_g, net_d, opt_g, opt_d, ema,
        )

    # ---- Training loop ----
    print(f"\nPhase 1 (warmup G-only): iters 0 → {args.warmup_iter}")
    print(f"Phase 2 (GAN):          iters {args.warmup_iter} → {args.total_iter}\n")

    # Infinite data iterator (wraps around epochs automatically)
    data_iter    = iter(train_loader)
    current_iter = start_iter
    epoch        = 0
    t0           = time.time()
    running_g    = 0.0
    running_d    = 0.0
    running_psnr = 0.0
    log_count    = 0

    while current_iter < args.total_iter:
        # Fetch next batch (wrap at epoch boundary)
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)
            # Step schedulers per epoch
            sched_g.step()
            if current_iter >= args.warmup_iter:
                sched_d.step()

        current_iter += 1
        lr_img = batch["lr"].to(device, non_blocking=True)
        hr_img = batch["hr"].to(device, non_blocking=True)

        gan_phase = current_iter > args.warmup_iter

        # ==================================================================
        # Generator step
        # ==================================================================
        net_g.train()

        # If in GAN phase, first get discriminator predictions on fake
        disc_fake_pred = None
        if gan_phase:
            with autocast(device_type="cuda", enabled=use_amp):
                sr = net_g(lr_img)
                # Discriminator forward on fake (no grad for D during G step)
                for p in net_d.parameters():
                    p.requires_grad = False
                disc_fake_pred = net_d(sr)

            # Compute generator loss with adversarial term
            opt_g.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                g_loss, g_log = g_criterion(sr, hr_img, disc_fake_pred)

            scaler_g.scale(g_loss).backward()
            # Gradient clipping for generator
            scaler_g.unscale_(opt_g)
            nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
        else:
            # Warmup: G-only, no adversarial loss
            opt_g.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                sr = net_g(lr_img)
                g_loss, g_log = g_criterion(sr, hr_img, disc_fake_pred=None)

            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

        # Update EMA
        ema.update(net_g)

        # PSNR on this batch (for logging)
        with torch.no_grad():
            batch_psnr = calc_psnr(sr.float().clamp(0, 1), hr_img.float())

        running_g    += g_log["g_total"]
        running_psnr += batch_psnr
        log_count    += 1

        # ==================================================================
        # Discriminator step (Phase 2 only)
        # ==================================================================
        d_log = {}
        if gan_phase:
            for p in net_d.parameters():
                p.requires_grad = True

            opt_d.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                # Real
                real_pred = net_d(hr_img)
                # Fake (detach generator output)
                fake_pred = net_d(sr.detach())
                d_loss, d_log = d_criterion(real_pred, fake_pred)

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(opt_d)
            nn.utils.clip_grad_norm_(net_d.parameters(), max_norm=1.0)
            scaler_d.step(opt_d)
            scaler_d.update()

            running_d += d_log["d_total"]

        # ==================================================================
        # Logging (every log_every iterations)
        # ==================================================================
        if current_iter % args.log_every == 0:
            elapsed = time.time() - t0
            avg_g    = running_g    / max(log_count, 1)
            avg_psnr = running_psnr / max(log_count, 1)
            avg_d    = running_d    / max(log_count, 1) if gan_phase else 0.0
            phase    = "GAN" if gan_phase else "warmup"
            cur_lr   = opt_g.param_groups[0]["lr"]

            parts = [
                f"[{phase}] iter {current_iter}/{args.total_iter}",
                f"G: {avg_g:.4f}",
            ]
            if gan_phase:
                parts.append(f"D: {avg_d:.4f}")
            parts += [
                f"PSNR: {avg_psnr:.2f}dB",
                f"LR: {cur_lr:.2e}",
                f"Time: {elapsed:.0f}s",
            ]

            # Add component losses from latest step
            comp = []
            if "g_l1" in g_log:
                comp.append(f"L1={g_log['g_l1']:.4f}")
            if "g_percep" in g_log:
                comp.append(f"Percep={g_log['g_percep']:.4f}")
            if "g_gan" in g_log:
                comp.append(f"GAN={g_log['g_gan']:.4f}")
            if comp:
                parts.append(f"({', '.join(comp)})")

            print("  ".join(parts))

            running_g    = 0.0
            running_d    = 0.0
            running_psnr = 0.0
            log_count    = 0
            t0 = time.time()

        # ==================================================================
        # Validation
        # ==================================================================
        if current_iter % args.val_every == 0:
            val_loss, val_psnr = validate(
                net_g, val_loader, g_criterion, device, use_amp,
            )
            print(
                f"\n  >> Validation @ iter {current_iter}: "
                f"Loss={val_loss:.6f}  PSNR={val_psnr:.2f}dB\n"
            )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint(
                    ckpt_dir / "YD_UPSCALE_anime_x4_best.pth",
                    net_g, net_d, opt_g, opt_d, ema,
                    current_iter, best_val_loss, args,
                )
                print(f"  >> New best val_loss={val_loss:.6f}")

                for inp in preview_inputs:
                    if inp.exists():
                        out = preview_dir / f"{inp.stem}_best.png"
                        save_preview(net_g, device, inp, out, use_amp)

        # ==================================================================
        # Periodic checkpoint
        # ==================================================================
        if current_iter % args.save_every == 0:
            _save_checkpoint(
                ckpt_dir / f"YD_UPSCALE_anime_x4_iter_{current_iter}.pth",
                net_g, net_d, opt_g, opt_d, ema,
                current_iter, best_val_loss, args,
            )
            for inp in preview_inputs:
                if inp.exists():
                    out = preview_dir / f"{inp.stem}_iter_{current_iter}.png"
                    save_preview(net_g, device, inp, out, use_amp)

    print("\nTraining complete.")


# ============================================================================
# Checkpoint save helper
# ============================================================================

def _save_checkpoint(
    path: Path,
    net_g: nn.Module,
    net_d: nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    ema: ModelEMA,
    iteration: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    """Save a full training checkpoint."""
    state = {
        "iteration":        iteration,
        "model_state_dict": net_g.state_dict(),
        "params_ema":       ema.state_dict(),
        "disc_state_dict":  net_d.state_dict(),
        "opt_g_state_dict": opt_g.state_dict(),
        "opt_d_state_dict": opt_d.state_dict(),
        "best_val_loss":    best_val_loss,
        "args":             vars(args),
    }
    torch.save(state, path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
