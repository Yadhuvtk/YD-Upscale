import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_comparisons(image_list, labels, save_path):
    """
    Plots a grid of images for easy subjective comparison.
    Expects images in HWC format [0, 255]
    """
    n = len(image_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
        
    for ax, img, title in zip(axes, image_list, labels):
        # Clip just in case
        vis_img = np.clip(img, 0, 255).astype(np.uint8)
        
        if vis_img.shape[0] < vis_img.shape[1] // 2: # Quick check if it needs resize to match target (like LR)
           vis_img = cv2.resize(vis_img, (image_list[-1].shape[1], image_list[-1].shape[0]), interpolation=cv2.INTER_NEAREST)
           
        ax.imshow(vis_img)
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
