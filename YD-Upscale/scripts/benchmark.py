import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.utils.config import parse_config  # pyre-ignore[21]
from yd_upscale.models.model_factory import create_model  # pyre-ignore[21]
from yd_upscale.data.dataloader_factory import create_dataset, build_dataloader  # pyre-ignore[21]
from yd_upscale.engine.evaluator import Evaluator  # pyre-ignore[21]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to benchmark YAML file')
    args = parser.parse_args()
    
    opt = parse_config(args.opt)
    print("Initializing model...")
    # NOTE: benchmark testing stubs
    model = create_model(opt['network_g']).cuda()
    
    print("Loading test dataset...")
    test_dataset = create_dataset(opt['datasets']['test'])
    test_loader = build_dataloader(test_dataset, opt['datasets']['test'], 'val')
    
    print("Evaluating...")
    from yd_upscale.utils.file_ops import ensure_dir  # pyre-ignore[21]
    save_dir = opt['benchmark'].get('output_dir', None)
    if save_dir: ensure_dir(save_dir)
    
    evaluator = Evaluator(opt['benchmark']['metrics'], device='cuda')
    results = evaluator.evaluate(model, test_loader, save_dir=save_dir)
    
    print(f"Benchmark Results: {results}")

if __name__ == '__main__':
    main()
