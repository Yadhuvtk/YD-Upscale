import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.data.dataset_paired import PairedDataset

def test_paired_dataset_init():
    opt = {
        'dataroot_hr': 'data/raw/train_hr',
        'patch_size': 256,
        'degradations': {}
    }
    dataset = PairedDataset(opt)
    assert len(dataset) >= 1
