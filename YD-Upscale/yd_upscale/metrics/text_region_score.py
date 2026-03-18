from yd_upscale.utils.registry import METRICS  # pyre-ignore[21]

@METRICS.register
def calculate_text_score(img1, img2):
    """Stub for text region score calculation"""
    return 0.0
