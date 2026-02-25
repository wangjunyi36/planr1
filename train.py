from argparse import ArgumentParser
import os

# MIG GPU 上 PyTorch CUDACachingAllocator 与 NVML 存在兼容问题，设置此变量可能缓解
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from callbacks import GPUMetricsCallback, DataStepTimingCallback
from datamodules import NuplanDataModule
from model import PlanR1
from utils import load_config

CURRENT_FILE_PATH = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_FILE_PATH)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train/pred.yaml')
    args = parser.parse_args()
    config = load_config(args.config)

    # Load the model from a checkpoint if provided
    if config['trainer']['ckpt_path']:
        print(f"Loading model from checkpoint: {config['trainer']['ckpt_path']}")
        model = PlanR1.load_from_checkpoint(config['trainer']['ckpt_path'], **config['model'])
    else:
        print("No checkpoint path provided, initializing a new model.")
        model = PlanR1(**config['model'])
    
    datamodule = NuplanDataModule(**config['datamodule'])
    model_checkpoint = ModelCheckpoint(**config['trainer']['ckpt'])
    lr_monitor = LearningRateMonitor(**config['trainer']['lr_monitor'])
    csv_logger = CSVLogger(**config['trainer']['csv_logger'])
    tb_cfg = dict(config['trainer'].get('tensorboard_logger', config['trainer']['csv_logger']))
    tb_cfg['version'] = csv_logger.version
    tensorboard_logger = TensorBoardLogger(**tb_cfg)
    trainer_kw = dict(
        strategy=config['trainer']['strategy'],
        devices=config['trainer']['devices'],
        accelerator=config['trainer']['accelerator'],
        callbacks=[model_checkpoint, lr_monitor, GPUMetricsCallback(), DataStepTimingCallback()],
        max_epochs=config['trainer']['max_epochs'],
        logger=[csv_logger, tensorboard_logger]
    )
    if 'log_every_n_steps' in config['trainer']:
        trainer_kw['log_every_n_steps'] = config['trainer']['log_every_n_steps']
    trainer = pl.Trainer(**trainer_kw)

    print("Starting training (first step may take 30s-2min for data loading)...")
    trainer.fit(model, datamodule)