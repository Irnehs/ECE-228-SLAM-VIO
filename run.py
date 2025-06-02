from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
import torch.nn as nn
import torch
import lightning.pytorch as pl
import argparse
from torch.utils.data import DataLoader, random_split
import os

from tools.data_processing_tools import IMUImageDataset, prep_combined_csv
from models.SLAMErrorPredictor import SLAMErrorPredictor
from tools.LitSLAMWrapper import LitSLAMWrapper

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    # Read cmd args, mainly to find config.yaml file
    parser = argparse.ArgumentParser(description='Model runner')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='config.yaml')
    parser.add_argument('--test',
                        help='flag to control running test set',
                        default=False)
    parser.add_argument('--train',
                        help='whether or not to train model',
                        default=True)

    args = parser.parse_args()
    # Open config file
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    tb_logger = TensorBoardLogger(save_dir=config['logging']['save_dir'])
    print(f"TensorBoard logs will be saved to: {tb_logger.log_dir}")

    model = SLAMErrorPredictor(**config['model'], seq_len=config['dataset']['seq_len'])

    # Define loss function from config
    loss_cfg = config["loss"]
    loss_class = getattr(nn, loss_cfg["name"])
    args_dict = loss_cfg.get("args") or {}
    loss_fnc = loss_class(**args_dict)

    seed_everything(config['experiment']['manual_seed'], True)

    lit_wrapper = LitSLAMWrapper(
        model,
        loss_fn=loss_fnc,
        lr=config['trainer']['LR'],
        weight_decay=config["experiment"]["weight_decay"],
        scheduler=config["experiment"]["scheduler"],
        scheduler_gamma=config["experiment"]["scheduler_gamma"],
        step_size=config["experiment"]["step_size"],
        validation_output_file=config["logging"]["validation_output_file"],
        test_output_file=config["logging"]["testing_output_file"],
    )
        
    model = SLAMErrorPredictor(**config['model'], seq_len=config['dataset']['seq_len'])


    train_data_path = config['dataset']['train_data']['data_path']
    train_combined_csv_path = config['dataset']['train_data']['combined_csv_path']
    train_vio_csv_path = config['dataset']['train_data']['vio_csv_path']
    prep_combined_csv(train_data_path, train_vio_csv_path, train_combined_csv_path)

    test_data_path = config['dataset']['train_data']['data_path']
    test_combined_csv_path = config['dataset']['train_data']['combined_csv_path']
    test_vio_csv_path = config['dataset']['train_data']['vio_csv_path']
    prep_combined_csv(test_data_path, test_vio_csv_path, test_combined_csv_path)

    H = config['dataset']['image_height']
    W = config['dataset']['image_width']

    cam0_path = os.path.join(train_data_path, 'cam0/data')
    cam1_path = os.path.join(train_data_path, 'cam1/data')

    # Split training data into train and val sets
    train_data = IMUImageDataset(
        csv_path=train_combined_csv_path,
        cam0_image_root=cam0_path,
        cam1_image_root=cam1_path,
        vio_predictions_path=train_vio_csv_path,
        seq_len=config['dataset']['seq_len'],
        prediction_len=config['model']['prediction_len'],
        H=H,
        W=W,
    )

    total_len = len(train_data)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(228)  # fix seed so split is the same every run

    train_dataset, val_dataset = random_split(
        train_data,
        [train_len, val_len],
        generator=generator
    )

    # Convert sets to dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataset']['train_batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataset']['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Dataloader for test set
    cam0_path = os.path.join(test_data_path, 'cam0/data')
    cam1_path = os.path.join(test_data_path, 'cam1/data')
    test_dataset = IMUImageDataset(
        csv_path=test_combined_csv_path,
        cam0_image_root=cam0_path,
        cam1_image_root=cam1_path,
        vio_predictions_path=test_vio_csv_path,
        seq_len=config['dataset']['seq_len'],
        prediction_len=config['model']['prediction_len'],
        H=H,
        W=W,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['dataset']['val_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Trainer
    if config["trainer"].get("gpus") is None:
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

    ### RUN TRAINING
    if args.train:
        trainer.fit(lit_wrapper, train_loader, val_loader)

    ### RUN TEST
    if args.test:
        if not args.train:
            # Load model from storage if not retraining
            state_dict = torch.load("final_model.pth", map_location="cpu")
            model.load_state_dict(state_dict)

        model.to(devices)

        # TODO put any test set functions in LitSLAMWrapper's pass blocks
        if config["trainer"]["gpus"] is None:
            if torch.backends.mps.is_available():
                accelerator = "mps"
                devices = 1
            else:
                accelerator = "cpu"
                devices = 1
        else:
            accelerator = "gpu"
            devices = config["trainer"]["gpus"]

        if args.test:
            trainer.test(lit_wrapper, test_loader)