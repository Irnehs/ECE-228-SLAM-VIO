from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
import yaml
import torch.nn as nn
import torch
import lightning.pytorch as pl
import argparse
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
                        default='')
    parser.add_argument('--train',
                        help='whether or not to train model',
                        default='')
    parser.add_argument('--mode',
                        help='train on either gt or vio data',
                        default='gt')

    args = parser.parse_args()
    args.train = bool(args.train)
    args.test = bool(args.test)
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

    if args.mode != 'gt' and args.mode != 'vio':
        assert TypeError, "Mode must either be 'gt' or 'vio"

    train_data_path = config['dataset']['train_data']['data_path']
    train_combined_csv_path = config['dataset']['train_data']['combined_csv_path']
    train_vio_csv_path = config['dataset']['train_data']['vio_csv_path']
    prep_combined_csv(train_data_path, train_vio_csv_path, train_combined_csv_path)

    test_data_path = config['dataset']['test_data']['data_path']
    test_combined_csv_path = config['dataset']['test_data']['combined_csv_path']
    test_vio_csv_path = config['dataset']['test_data']['vio_csv_path']
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
        batch_size=config['dataset']['test_batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        accelerator = "gpu"
        device = 'cuda'
        devices = 1
    else:
        accelerator = "cpu"
        device = 'cpu'
        devices = 1

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=config["trainer"]["max_epochs"],
        accelerator=accelerator,
        devices=devices,
    )

    if args.train:
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
            mode=args.mode,
        )
    else:
        lit_wrapper = LitSLAMWrapper.load_from_checkpoint(
            "final.ckpt",
            model=model,
            loss_fn=loss_fnc,
            lr=config['trainer']['LR'],
            weight_decay=config["experiment"]["weight_decay"],
            scheduler=config["experiment"]["scheduler"],
            scheduler_gamma=config["experiment"]["scheduler_gamma"],
            step_size=config["experiment"]["step_size"],
            validation_output_file=config["logging"]["validation_output_file"],
            test_output_file=config["logging"]["testing_output_file"],
            mode=args.mode,
        )

    ### RUN TRAINING
    if args.train:
        trainer.fit(lit_wrapper, train_loader, val_loader)

    ### RUN TEST
    if args.test:
        model = lit_wrapper.model
        model.to(device).eval()

        all_errors = []
        with torch.no_grad():
            error1 = []
            error2 = []
            error3 = []
            for idx, (x, y) in enumerate(test_loader):
                data = x['data']
                gt = y['ground_truth']
                vio = y['vio']

                data_device = []
                for x in data:
                    data_device.append(x.to(device))  # shape: [1, …]
                gt = gt.to(device)  # shape: [1, …]
                vio = vio.to(device)

                out = model(data_device)  # forward pass

                per_sample_loss = loss_fnc(out, vio)
                per_sample_loss2 = loss_fnc(vio, gt)
                per_sample_loss3 = loss_fnc(out, gt)

                error1.append(per_sample_loss.cpu())
                error2.append(per_sample_loss2.cpu())
                error3.append(per_sample_loss3.cpu())

            x = np.arange(len(error1)) * config['dataset']['test_batch_size']  # array([0, 16, 32, 48, …])

            plt.figure(figsize=(8, 3))
            plt.plot(x, error1, linestyle="-", markersize=3)
            plt.plot(x, error2, linestyle="-", markersize=3)
            plt.plot(x, error3, linestyle="-", markersize=3)
            plt.legend(['Model vs. VIO Pose', 'VIO vs. Ground Truth Pose', 'Model vs. Ground Truth Pose'])
            plt.xlabel("Test Sample Index (sequential)")
            plt.ylabel("MSE Error")
            if "medium" in config['dataset']['test_data']['data_path']:
                str = "MSE Test Pose Error on Medium Dataset"
            else:
                str = "MSE Test Pose Error on Hard Dataset"
            plt.title(str)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

            df = pd.DataFrame({
                'Model vs. VIO Pose':         error1,
                'VIO vs. Ground Truth Pose':  error2,
                'Model vs. Ground Truth Pose':error3,
            })

            df.to_csv('test_output_' + args.mode + '.csv', index=True)  # ‘index=False’ omits the row numbers column