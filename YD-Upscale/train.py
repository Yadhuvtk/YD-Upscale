from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler

from yd_upscale.data.dataloader_factory import build_train_val_dataloaders
#from yd_upscale.models.rrdbnet import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


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


def main():
    project_root = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale")
    manifest_dir = project_root / "data" / "manifests"
    checkpoint_dir = project_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1
    num_workers = 2
    scale = 4
    patch_size_lr = 128
    epochs = 1
    lr_rate = 2e-4
    use_amp = torch.cuda.is_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_train_val_dataloaders(
        manifest_dir=manifest_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        scale=scale,
        patch_size_lr=patch_size_lr,
    )

    model = SRVGGNetCompact(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_conv=32,
    upscale=scale,
    act_type="prelu",
).to(device)

    criterion = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=lr_rate, betas=(0.9, 0.99))
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Batch size:    {batch_size}")
    print(f"LR patch:      {patch_size_lr}")
    print(f"HR patch:      {patch_size_lr * scale}")

    for epoch in range(1, epochs + 1):
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
                print(
                    f"Epoch [{epoch}/{epochs}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"Train Loss: {avg_loss:.6f}"
                )

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        elapsed = time.time() - start_time

        print(
            f"\nEpoch {epoch} done | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Time: {elapsed/60:.2f} min\n"
        )

        #ckpt_path = checkpoint_dir / f"yd_upscale_rrdb_x4_epoch_{epoch}.pth"
        ckpt_path = checkpoint_dir / f"YD_UPSCALE_x4_epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()