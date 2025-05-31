from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
import torch.nn as nn
import lightning.pytorch as pl
import argparse
from torch.utils.data import DataLoader
import os

from tools.data_processing_tools import IMUImageDataset, prep_combined_csv
from models.SLAMErrorPredictor import SLAMErrorPredictor
from tools.LitSLAMWrapper import LitSLAMWrapper

if __name__ == "__main__":
    # Read cmd args, mainly to find config.yaml file
    parser = argparse.ArgumentParser(description='Model runner')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config.yaml')

    # Open config file
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger =  TensorBoardLogger(save_dir=config['logging']['save_dir'])

    # Define loss function from config
    loss_cfg = config["loss"]
    loss_class = getattr(nn, loss_cfg["name"])
    args_dict = loss_cfg.get("args") or {}
    loss_fnc  = loss_class(**args_dict)

    seed_everything(config['experiment']['manual_seed'], True)
        
    model = SLAMErrorPredictor(**config['model'], seq_len=config['dataset']['seq_len'])
    lit_wrapper = LitSLAMWrapper(
        model,
        loss_fn=loss_fnc,
        lr=config['trainer']['LR'],
        weight_decay=config["experiment"]["weight_decay"],
        scheduler=config["experiment"]["scheduler"],
        scheduler_gamma=config["experiment"]["scheduler_gamma"],
        step_size=config["experiment"]["step_size"],
    )

    data_path = config['dataset']['data_path']
    csv_path = config['dataset']['csv_path']
    prep_combined_csv(data_path, csv_path)

    cam0_path = os.path.join(data_path, 'cam0/data')
    cam1_path = os.path.join(data_path, 'cam1/data')
    train_data = IMUImageDataset(
        csv_path=csv_path,
        cam0_image_root=cam0_path,
        cam1_image_root=cam1_path,
        seq_len=config['dataset']['seq_len']
    )
    # TODO get different flight data for validation
    val_data = IMUImageDataset(
        csv_path=csv_path,
        cam0_image_root=cam0_path,
        cam1_image_root=cam1_path,
        seq_len=config['dataset']['seq_len']
    )
    train_loader = DataLoader(train_data, batch_size=config['dataset']['train_batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    val_loader   = DataLoader(val_data,   batch_size=config['dataset']['val_batch_size'], num_workers=config['dataset']['num_workers'])
    
    # Trainer
    if config["trainer"]["gpus"] is None:
        accelerator = "cpu"
        devices = 1
    else:
        accelerator = "gpu"
        devices = config["trainer"]["gpus"]

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(lit_wrapper, train_loader, val_loader)