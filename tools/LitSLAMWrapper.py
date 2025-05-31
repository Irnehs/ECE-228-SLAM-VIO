import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class LitSLAMWrapper(pl.LightningModule):
    def __init__(self, model, loss_fn=nn.MSELoss, lr=1e-3, weight_decay=0, scheduler=None, scheduler_gamma=None, step_size=1):
        super(LitSLAMWrapper, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model','loss_fn'])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
            x, y = batch
            y_pred = self(x)
            loss = self.loss_fn(y_pred, y)
            self.log("train_loss", loss)
            return loss
        
    def validation_step(self, batch, dataloader_idx=0):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        self.log(f"val_loss_loader{dataloader_idx}", val_loss)  # In case we want to split validation into easy/hard, we can use the index with multiple loaders

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