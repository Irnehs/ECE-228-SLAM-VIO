import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
import pandas as pd

class LitSLAMWrapper(pl.LightningModule):
    def __init__(self, model, validation_output_file : str, test_output_file : str, loss_fn=nn.MSELoss, lr=1e-3, weight_decay=0, scheduler=None, scheduler_gamma=None, step_size=1):
        super(LitSLAMWrapper, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model','loss_fn'])
        self.validation_output_file = validation_output_file
        self.test_output_file = test_output_file
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, batch):
        return self.model(batch['data'])

    def training_step(self, batch):
            x, y = batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred, y['ground_truth'])  # Assuming y contains ground truth poses
            self.log("train_loss", loss)
            return loss

    def on_train_end(self) -> None:
        torch.save(self.model.state_dict(), "final_model.pth")
        
    def validation_step(self, batch, dataloader_idx=0):
        x, y = batch
        timestamps = x["timestamp"]  # shape: (B, 10)

        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y['ground_truth'])
        self.log("val_loss_loader", val_loss)

        keys = ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']

        B = y_pred.shape[0]
        pred_seq = y_pred.view(B, 10, 7)
        vio_seq = y['vio'].view(B, 10, 7)
        gt_seq = y['ground_truth'].view(B, 10, 7)

        # Compare loss curves
        model_vs_vio_l2 = torch.linalg.norm(pred_seq - vio_seq, dim=2).mean()
        vio_vs_gt_l2 = torch.linalg.norm(vio_seq - gt_seq, dim=2).mean()
        # TensorBoard plot overlays
        self.logger.experiment.add_scalars(
            "l2_compare/combined",
            {
                "model_vs_vio": model_vs_vio_l2,
                "vio_vs_gt": vio_vs_gt_l2
            },
            global_step=self.global_step
        )

        for b in range(B):
            for i in range(10):
                row = {
                    "epoch": self.current_epoch,
                    "timestep": i,
                    "timestamp": timestamps[b, -1].item()                    
                }
                for name, tensor in [('pose_pred', pred_seq[b, i]),
                                    ('vio', vio_seq[b, i]),
                                    ('gt', gt_seq[b, i])]:
                    for k, key in enumerate(keys):
                        row[f"{name}_{key}"] = tensor[k].item()
                self.validation_outputs.append(row)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        timestamps = x["timestamp"]  # shape: (B, seq_len)

        y_pred = self(x)  # shape: (B, 70)
        keys = ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']

        B = y_pred.shape[0]
        pred_seq = y_pred.view(B, 10, 7)
        vio_seq = y["vio"].view(B, 10, 7)
        gt_seq = y["ground_truth"].view(B, 10, 7)

        # Compare loss curves
        model_vs_vio_l2 = torch.linalg.norm(pred_seq - vio_seq, dim=2).mean()
        vio_vs_gt_l2 = torch.linalg.norm(vio_seq - gt_seq, dim=2).mean()
        # TensorBoard plot overlays
        self.logger.experiment.add_scalars(
            "l2_compare/combined",
            {
                "model_vs_vio": model_vs_vio_l2,
                "vio_vs_gt": vio_vs_gt_l2
            },
            global_step=self.global_step
        )

        for b in range(B):
            for i in range(10):
                row = {
                    "epoch": self.current_epoch,
                    "timestep": i,
                    "timestamp": timestamps[b, -1].item()                
                }
                for name, tensor in [('pose_pred', pred_seq[b, i]),
                                    ('vio', vio_seq[b, i]),
                                    ('gt', gt_seq[b, i])]:
                    for k, key in enumerate(keys):
                        row[f"{name}_{key}"] = tensor[k].item()
                self.test_outputs.append(row)

    def on_validation_epoch_end(self):
        if self.validation_outputs:
            df = pd.DataFrame(self.validation_outputs)
            df.to_csv(self.validation_output_file, index=False)
            self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if self.test_outputs:
            df = pd.DataFrame(self.test_outputs)
            df.to_csv(self.test_output_file, index=False)
            self.test_outputs.clear()

    def test_epoch_end(self, outputs):
        # called at the end of testing across all batches
        pass

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        if not self.hparams.scheduler:
            return opt

        try:
            schedule_class = getattr(lr_scheduler, self.hparams.scheduler)
        except AttributeError:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        sched = schedule_class(opt, gamma=self.hparams.scheduler_gamma)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",   # call scheduler.step() every epoch
                "frequency": self.hparams.step_size,
                "name": self.hparams.scheduler  # logs under this name in tensorboard
            }
        }