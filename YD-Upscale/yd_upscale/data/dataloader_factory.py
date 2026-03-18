from torch.utils.data import DataLoader
from yd_upscale.utils.registry import DATASETS

def create_dataset(dataset_opt):
    dataset_type = dataset_opt['type']
    dataset_class = DATASETS.get(dataset_type)
    if dataset_class is None:
        raise ValueError(f"Dataset {dataset_type} not found in registry.")
    
    return dataset_class(dataset_opt)

def build_dataloader(dataset, dataset_opt, phase):
    if phase == 'train':
        return DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt.get('use_shuffle', True),
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            drop_last=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=dataset_opt.get('batch_size', 1),
            shuffle=False,
            num_workers=dataset_opt.get('num_workers', 1),
            pin_memory=True
        )
