import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.utils.config import parse_config
from yd_upscale.models.model_factory import create_model
from yd_upscale.engine.inferencer import Inferencer
from yd_upscale.utils.file_ops import ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to inference config YAML file.')
    parser.add_argument('-input', type=str, default=None, help='Override input path (file or folder)')
    parser.add_argument('-output', type=str, default=None, help='Override output folder')
    args = parser.parse_args()
    
    opt = parse_config(args.opt)
    
    model = create_model(opt['network_g']).cuda()
    inferencer = Inferencer(model, use_half=opt['infer'].get('half', False))
    
    in_path = Path(args.input if args.input else opt['infer']['input_dir'])
    out_dir = Path(args.output if args.output else opt['infer']['output_dir'])
    ensure_dir(out_dir)
    
    if in_path.is_file():
        img_paths = [in_path]
    else:
        img_paths = [p for p in in_path.glob('*.*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
    
    for img_path in img_paths:
        if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
            out_path = out_dir / f"{img_path.stem}_out.png"
            print(f"Processing {img_path.name}...")
            inferencer.infer(str(img_path), str(out_path))
            
    print("Done.")

if __name__ == '__main__':
    main()
