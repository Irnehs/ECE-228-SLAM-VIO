from argparse import ArgumentParser
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
import torch.nn as nn

from models.SLAMErrorPredictor import SLAMErrorPredictor
from experiment import Experiment

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

    # For reproducibility
    seed_everything(config['experiment']['manual_seed'], True)
        
    model = SLAMErrorPredictor(**config['model'])
    experiment = Experiment(
        loss_fnc=loss_fnc,
        lr=config['trainer']['LR'],
        weight_decay=config["experiment"]["weight_decay"],
        scheduler=config["experiment"]["scheduler"],
        scheduler_gamma=config["experiment"]["scheduler_gamma"],
        step_size=config["experiment"]["step_size"],
    )
    
    # TODO slot in data pipelining into train_data and val_data
    train_data = DummyIMUDataset(seq_len=20, num_samples=2000)  # TODO
    val_data   = DummyIMUDataset(seq_len=20, num_samples=500)  # TODO
    train_loader = DataLoader(train_data, batch_size=config['dataset']['train_batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    val_loader   = DataLoader(val_data,   batch_size=config['dataset']['train_batch_size'], num_workers=config['dataset']['num_workers'])
    
    # Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config["trainer"]["max_epochs"],
        accelerator="auto"
        devices=config["trainer"]["gpus"],
    )
    trainer.fit(model, train_loader, val_loader)