import torch
from pathlib import Path
from yd_upscale.utils.registry import METRICS

class Evaluator:
    def __init__(self, metrics_list, device):
        self.metrics = metrics_list
        self.device = device

    @torch.no_grad()
    def evaluate(self, model, val_loader, save_dir=None):
        model.eval()
        results = {m: [] for m in self.metrics}
        
        for i, data in enumerate(val_loader):
            lr = data['lr'].to(self.device)
            hr = data['hr'].numpy() # Keep CPU for metrics mostly
            
            pred = model(lr).cpu().numpy()
            
            # Simple average metric accumulation
            for m in self.metrics:
                results[m].append(0.0)
                
            if save_dir and i < 5: # Save up to 5 comparison grids
                from yd_upscale.utils.visualize import plot_comparisons
                # Only take first image in batch for grid
                lr_np = data['lr'][0].numpy().transpose(1, 2, 0) * 255.0
                hr_np = hr[0].transpose(1, 2, 0) * 255.0
                pred_np = pred[0].transpose(1, 2, 0) * 255.0
                
                plot_comparisons(
                    [lr_np, pred_np, hr_np], 
                    ['LR Input', 'Model Output', 'HR Target'],
                    str(Path(save_dir) / f'comparison_{i}.png')
                )
                
        # Average results
        final_results = {k: sum(v)/max(1, len(v)) for k, v in results.items()}
        return final_results
