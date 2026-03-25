from pathlib import Path

folder = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_lr_x4")

files = sorted(folder.glob("*.png"))

for i, file in enumerate(files, start=10):
    file.rename(folder / f"__tmp__{i}.png")

tmp_files = sorted(folder.glob("__tmp__*.png"))

for i, file in enumerate(tmp_files, start=1):
    file.rename(folder / f"{i}.png")

print("Done")