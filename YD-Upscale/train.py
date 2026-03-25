from __future__ import annotations

from pathlib import Path
import sys
import time
import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms.functional as TF

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yd_upscale.models.rrdbnet import RRDBNet
from yd_upscale.data.dataloader_factory import build_train_val_dataloaders
from yd_upscale.models.srvggnet import SRVGGNetCompact
from yd_upscale.losses.illustration_loss import IllustrationLoss


def validate(model, val_loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            with autocast(device_type="cuda", enabled=use_amp):
                sr = model(lr)
                loss = criterion(sr, hr)

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


def save_epoch_preview(model, device, input_path: Path, output_path: Path, use_amp=True):
    model.eval()

    if not input_path.exists():
        print(f"[Preview] Input not found: {input_path}")
        return

    img = Image.open(input_path).convert("RGB")
    lr_tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            sr_tensor = model(lr_tensor).clamp(0, 1)

    sr_img = TF.to_pil_image(sr_tensor.squeeze(0).cpu())
    sr_img.save(output_path)
    print(f"[Preview] Saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size-lr", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--edge-weight", type=float, default=0.3)
    parser.add_argument("--online-degrade", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    manifest_dir = project_root / "data" / "manifests"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = checkpoint_dir / "epoch_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    preview_inputs = [
        project_root / "Testing" / "Test.jpg",
        project_root / "Testing" / "Tests.jpg",
        project_root / "Testing" / "testtt.png",
    ]

    use_amp = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Project root: {project_root}")
    print(f"Manifest dir: {manifest_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Online degrade: {args.online_degrade}")

    train_loader, val_loader = build_train_val_dataloaders(
        manifest_dir=manifest_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        patch_size_lr=args.patch_size_lr,
        online_degrade=args.online_degrade,
    )

    model = RRDBNet(
    in_nc=3,
    out_nc=3,
    nf=64,
    nb=23,
    gc=32,
    scale=args.scale,
).to(device)

    criterion = IllustrationLoss(edge_weight=args.edge_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Batch size:    {args.batch_size}")
    print(f"LR patch:      {args.patch_size_lr}")
    print(f"HR patch:      {args.patch_size_lr * args.scale}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, start=1):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if step % 25 == 0:
                avg_loss = running_loss / step
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.6f} "
                    f"LR: {current_lr:.8f}"
                )

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        elapsed = time.time() - start_time
        scheduler.step()

        print(
            f"\nEpoch {epoch} done | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed/60:.2f} min\n"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        epoch_ckpt_path = checkpoint_dir / f"YD_UPSCALE_x4_epoch_{epoch}.pth"
        torch.save(state, epoch_ckpt_path)
        print(f"Saved checkpoint: {epoch_ckpt_path}")

        for input_path in preview_inputs:
            if input_path.exists():
                stem = input_path.stem
                epoch_preview_path = preview_dir / f"{stem}_epoch_{epoch}.png"
                save_epoch_preview(
                    model=model,
                    device=device,
                    input_path=input_path,
                    output_path=epoch_preview_path,
                    use_amp=use_amp,
                )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = checkpoint_dir / "YD_UPSCALE_x4_best.pth"
            torch.save(state, best_ckpt_path)
            print(f"Saved best checkpoint: {best_ckpt_path}")

            for input_path in preview_inputs:
                if input_path.exists():
                    stem = input_path.stem
                    best_preview_path = preview_dir / f"{stem}_best.png"
                    save_epoch_preview(
                        model=model,
                        device=device,
                        input_path=input_path,
                        output_path=best_preview_path,
                        use_amp=use_amp,
                    )


if __name__ == "__main__":
    main()