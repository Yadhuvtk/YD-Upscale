import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.utils.config import parse_config
from yd_upscale.models.model_factory import create_model
from yd_upscale.data.dataloader_factory import create_dataset, build_dataloader
from yd_upscale.engine.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    
    opt = parse_config(args.opt)
    model = create_model(opt['network_g']).cuda()
    
    val_dataset = create_dataset(opt['datasets']['val'])
    val_loader = build_dataloader(val_dataset, opt['datasets']['val'], 'val')
    
    evaluator = Evaluator(["PSNR", "SSIM"], device='cuda')
    results = evaluator.evaluate(model, val_loader)
    print("Validation Results:", results)

if __name__ == '__main__':
    main()
