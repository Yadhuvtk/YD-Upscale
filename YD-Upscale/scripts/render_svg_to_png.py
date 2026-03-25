from pathlib import Path
import subprocess
import shutil

# -------- SETTINGS --------
input_folder = Path(r"E:\Yadhu Projects\CDR SVG")
output_folder = Path(r"E:\Yadhu Projects\YD-Upscale\YD-Upscale\data\rendered_hr")
inkscape_path = r"C:\Program Files\Inkscape\bin\inkscape.exe"
size = 2048
# --------------------------

def main():
    if not input_folder.exists():
        print(f"Input folder not found: {input_folder}")
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    svg_files = sorted(input_folder.glob("*.svg"))

    if not svg_files:
        print("No SVG files found.")
        return

    print(f"Found {len(svg_files)} SVG files")

    for i, svg_file in enumerate(svg_files, start=1):
        png_file = output_folder / f"{svg_file.stem}.png"

        cmd = [
            inkscape_path,
            str(svg_file),
            "--export-type=png",
            f"--export-filename={png_file}",
            f"--export-width={size}",
            "--export-background=white",
            "--export-background-opacity=1.0",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                print(f"[{i}/{len(svg_files)}] Converted: {svg_file.name} -> {png_file.name}")
            else:
                print(f"[{i}/{len(svg_files)}] Failed: {svg_file.name}")
                print(result.stderr.strip() or result.stdout.strip())

        except Exception as e:
            print(f"[{i}/{len(svg_files)}] Error: {svg_file.name} | {e}")

    print("Done.")

if __name__ == "__main__":
    main()