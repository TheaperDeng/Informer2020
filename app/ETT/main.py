import sys
sys.path.append('../../models')
sys.path.append('../../data')
sys.path.append('../..')

from data_loader import Dataset_Custom
from pl_model import Informer_pl
from torch.utils.data import DataLoader
import numpy as np
from bigdl.nano.pytorch.trainer import Trainer

dataset = Dataset_Custom("/home/junweid/Informer2020/downloads/ETT-small", flag='train', size=None, 
                         features='S', data_path='ETTh1.csv', 
                         target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Informer_pl(enc_in=1, dec_in=1, c_out=1, seq_len=24*4*4, label_len=24*4, out_len=24*4)

trainer = Trainer(max_epochs = 5)
trainer.fit(model, train_dataloader)