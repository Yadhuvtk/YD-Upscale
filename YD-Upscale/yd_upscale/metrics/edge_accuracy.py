from yd_upscale.utils.registry import METRICS  # pyre-ignore[21]

@METRICS.register
def calculate_edge_accuracy(img1, img2):
    """Stub for edge accuracy score"""
    return 0.0
