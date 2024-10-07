import os
import random
import pickle
import logging
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from dataset import *
from model.CNN3D import *
from trainer import Trainer
from utils import seed_everything

seed_everything(42)
''' CNN Embedding Dimension 계산부터 다시 하면 됨'''
# == arguments & parameters == #
hidden_dims = [[32, 64], [16, 32], [32, 48]]
kernel_sizes = [[(5,5,5), (3,3,3)], [(3,3,3), (3,3,3)]]
drop_p = [0.0, 0.1]
fc_hidden = [256, 512]
skip_frames = [500, 250, 100]
output_dim=1
# train
batch_size = [2, 4]
learning_rate = 1e-4
T_max = 30
eta_min = 1e-5
epochs = 50
patience = 5
log_interval = 10
device = 'cuda'
data_path = '../workspace/frames/frames'
finetune = True # If finetune, checkpoint will not be saved

# ============================ #d
   
with open('../workspace/processed_data/train_split.pickle', 'rb') as f:
    train_list_label = pickle.load(f)
f.close()

with open('../workspace/processed_data/valid_split.pickle', 'rb') as f:
    valid_list_label = pickle.load(f)
f.close()

with open('../workspace/processed_data/test_split.pickle', 'rb') as f:
    test_list_label = pickle.load(f)
f.close()

train_list = train_list_label[0]
val_list = valid_list_label[0]
test_list = test_list_label[0]

train_label = np.array(train_list_label[1])
val_label = np.array(valid_list_label[1])
test_label = np.array(test_list_label[1])

median = np.median(train_label, axis=0)
iqr = np.quantile(train_label, 0.75, axis=0) - np.quantile(train_label, 0.25, axis=0)

train_label = (train_label-median)/iqr
val_label = (val_label-median)/iqr
test_label = (test_label-median)/iqr

hd_col = []
ks_col = []
fc_col = []
do_col = []
bs_col = []
sf_col = []
best_loss_col = []
ang_freq = [1, 3, 10, 30, 100]

for i, freq in enumerate(ang_freq):
    for hd in hidden_dims:
        for ks in kernel_sizes: 
            for fc in fc_hidden:
                for do in drop_p:
                    for bs in batch_size:
                        for sf in skip_frames:
                            hd_col.append(str(hd))
                            ks_col.append(str(ks))
                            fc_col.append(str(fc))
                            do_col.append(str(do))
                            bs_col.append(str(bs))
                            sf_col.append(str(sf))
                            save_model_path = f'./3dcnn/single/{freq}/weights/hd{hd}_ks{ks}_fc{fc}_do{do}_bs{bs}_sf{sf}/'
                            save_pred_path = f'./3dcnn/single/{freq}/predictions/hd{hd}_ks{ks}_fc{fc}_do{do}_bs{bs}_sf{sf}/'
                            log_path = f'./3dcnn/single/{freq}/log/hd{hd}_ks{ks}_fc{fc}_do{do}_bs{bs}_sf{sf}/'
                            for paths in [log_path, save_model_path, save_pred_path]:
                                os.makedirs(paths, exist_ok=True)

                            # logger
                            logging.basicConfig(level=logging.INFO, filemode='w', format="%(message)s")
                            logger = logging.getLogger()
                            logger.addHandler(logging.FileHandler(log_path + '/log.log'))
                            logger.info(f'fc hidden: {fc}, dropout: {do}, batch size: {bs}')

                            begin_frame, end_frame, skip_frame = 0, 4999, sf
                            selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
                            t_dim = len(selected_frames)

                            params = {'batch_size': bs, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}
                            test_params = {'batch_size': bs, 'shuffle': False, 'num_workers': 40, 'pin_memory': True}

                            transform = transforms.Compose([transforms.Resize([128, 128]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.5], std=[0.5])])

                            train_set, valid_set, test_set = CustomDataset(data_path, train_list, train_label[:, i], selected_frames, transform=transform, cnn3d=True), \
                                                            CustomDataset(data_path, val_list, val_label[:, i], selected_frames, transform=transform, cnn3d=True), \
                                                            CustomDataset(data_path, test_list, test_label[:, i], selected_frames, transform=transform, cnn3d=True)
                                                            
                            train_loader = DataLoader(train_set, **params)
                            valid_loader = DataLoader(valid_set, **test_params)
                            test_loader = DataLoader(test_set, **test_params)

                            model = CNN3D(hidden_dim=hd, kernel_size=ks, fc_hidden=fc, drop_p=do, t_dim=t_dim, output_dim=output_dim).to(device)
                            # model = CNN3D(output_dim=output_dim).to(device)

                            criterion = nn.L1Loss()
                            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

                            trainer = Trainer(train_loader, valid_loader, test_loader, 
                                            model, optimizer, criterion, 
                                            epochs, scheduler, patience, 
                                            save_model_path, logger, log_interval, device)

                            best_loss = trainer.train()
                            best_loss_col.append(best_loss)
                            
                            logging.shutdown()

                            result = pd.DataFrame({'hidden dim': hd_col, 
                                                'kernel size': ks_col, 
                                                'fc_hidden': fc_col, 
                                                'dropout': do_col, 
                                                'batch_size': bs_col, 
                                                'skip frames': sf_col, 
                                                'best_loss': best_loss_col})
                            result.to_csv(f'finetune_result/3dcnn_single_{freq}.csv', index=None)

                            true, pred = trainer.predict()
                            true = true*np.array(iqr[i]) + np.array(median[i])
                            pred = pred*np.array(iqr[i]) + np.array(median[i])
                            with open(os.path.join(save_pred_path, f'convlstm_single_{freq}.pickle'), 'wb') as f:
                                pickle.dump([true, pred], f)
                            f.close()