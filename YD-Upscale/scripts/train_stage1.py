import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from torch.optim import Adam  # pyre-ignore[21]
from yd_upscale.utils.config import parse_config  # pyre-ignore[21]
from yd_upscale.utils.logger import setup_logger  # pyre-ignore[21]
from yd_upscale.utils.seed import set_random_seed  # pyre-ignore[21]
from yd_upscale.models.model_factory import create_model  # pyre-ignore[21]
from yd_upscale.data.dataloader_factory import create_dataset, build_dataloader  # pyre-ignore[21]
from yd_upscale.losses.loss_factory import build_loss  # pyre-ignore[21]
from yd_upscale.engine.trainer import Trainer  # pyre-ignore[21]
from yd_upscale.engine.checkpoint import save_checkpoint, load_checkpoint  # pyre-ignore[21]
from yd_upscale.engine.scheduler_factory import build_scheduler  # pyre-ignore[21]
from torch.utils.tensorboard import SummaryWriter  # pyre-ignore[21]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')
    args = parser.parse_args()
    
    opt = parse_config(args.opt)
    logger = setup_logger('train', log_file='logs/train/train_stage1.log')
    logger.info(f"Loaded config: {opt['name']}")
    
    set_random_seed(1234)
    
    # Model
    model = create_model(opt['network_g']).cuda()
    
    # Optimizer
    optim_opt = opt['train']['optim_g']
    optimizer = Adam(model.parameters(), lr=optim_opt['lr'], betas=optim_opt.get('betas', (0.9, 0.999)))
    
    # Scheduler
    scheduler = build_scheduler(opt['train']['scheduler'], optimizer)
    
    # Losses
    losses = {}
    for loss_name, loss_opt in opt['train']['losses'].items():
        losses[loss_name] = build_loss(loss_opt).cuda()
        
    # Data
    train_dataset = create_dataset(opt['datasets']['train'])
    train_loader = build_dataloader(train_dataset, opt['datasets']['train'], 'train')
    
    trainer = Trainer(model, optimizer, losses, use_amp=opt.get('use_amp', False), resume_state=args.resume)
    
    # Tensorboard Logging
    use_tb = opt.get('logger', {}).get('use_tb_logger', False)
    tb_writer = None
    if use_tb:
        tb_writer = SummaryWriter(log_dir=f"logs/tensorboard/{opt['name']}")
    
    logger.info("Starting training loop...")
    epochs = 1
    global_step = trainer.start_epoch * len(train_loader)
    
    for epoch in range(trainer.start_epoch, epochs):
        for i, data in enumerate(train_loader):
            loss_dict = trainer.train_step(data)
            
            if i % opt['logger']['print_freq'] == 0:  # pyre-ignore
                loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                logger.info(f"Epoch {epoch} | Iter {i} | {loss_str}")
                
                if tb_writer is not None:
                    for k, v in loss_dict.items():
                        tb_writer.add_scalar(f'losses/{k}', v, global_step)  # pyre-ignore
            global_step += 1
        
        # Scheduler step at end of epoch
        scheduler.step()
        save_checkpoint(model, optimizer, epoch, f"checkpoints/pretrain/net_g_{epoch}.pth")

    if tb_writer is not None:
        tb_writer.close()  # pyre-ignore
    
    logger.info("Training complete.")

if __name__ == '__main__':
    main()
