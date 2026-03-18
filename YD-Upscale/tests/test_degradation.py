import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from yd_upscale.data.degradations import degrade_image
import numpy as np

def test_degrade_image():
    hr = np.zeros((256, 256, 3), dtype=np.uint8)
    opt = {'blur_prob': 0, 'noise_prob': 0, 'jpeg_prob': 0}
    lr = degrade_image(hr, opt, scale=4)
    assert lr.shape == (64, 64, 3)
