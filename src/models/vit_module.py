# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, List

import torch
from pytorch_lightning import LightningModule


class ViTLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        pretrained_path: str,
        lr: float = 0.001,
        weight_decay: float = 0.005
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        if len(pretrained_path) > 0:
            self.load_mae_weights(pretrained_path)

    def load_mae_weights(self, pretrained_path):
        checkpoint = torch.load(pretrained_path)

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint['state_dict']
        state_dict = self.state_dict()
        checkpoint_keys = list(checkpoint_model.keys())
        for k in checkpoint_keys:
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def forward(self, x):
        return self.net.predict(x)

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, _ = self.net.forward(x, y)
        for var in loss_dict.keys():
            self.log("train/" + var, loss_dict[var], on_step=True, on_epoch=False, prog_bar=True)
        return loss_dict

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, _ = self.net.forward(x, y)
        for var in loss_dict.keys():
            self.log("val/" + var, loss_dict[var], on_step=False, on_epoch=True, prog_bar=False)
        return loss_dict

    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log(
    #         "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
    #     )

    #     self.val_acc.reset()  # reset val accuracy for next epoch

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, _ = self.net.forward(x, y)
        for var in loss_dict.keys():
            self.log("test/" + var, loss_dict[var], on_step=False, on_epoch=True)
        return loss_dict

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
