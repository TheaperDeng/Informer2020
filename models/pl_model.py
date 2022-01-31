import pytorch_lightning as pl
from model import Informer
import torch


class Informer_pl(pl.LightningModule):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, **kwargs):
        super().__init__()
        self.out_len = out_len
        self.label_len = label_len
        self.model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, **kwargs)
        self.loss = torch.nn.MSELoss()

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float()
        batch_y_mark = batch_y_mark.float()
        dec_inp = torch.zeros([batch_y.shape[0], self.out_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float()

        y_hat = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss = self.loss(y_hat, batch_y[:, -self.out_len:, :])

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
