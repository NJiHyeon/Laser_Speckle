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
from model.CNNRNN import *
from trainer import Trainer
from utils import seed_everything

seed_everything(42)
''' CNN Embedding Dimension 계산부터 다시 하면 됨'''
# == arguments & parameters == #
# cnn
hidden_dims = [[32, 64], [16, 32], [32, 48]]
kernel_sizes = [[(5,5,5), (3,3,3)], [(3,3,3), (3,3,3)]]
drop_p_cnn = [0.0, 0.1]
output_dim_cnn = [512, 256]
# lstm
hidden_dim = [128, 256]
num_layers = [1, 3]
output_dim = 1
drop_p_lstm = [0.0]
# train
batch_size = [2]
learning_rate = 1e-4
T_max = 30
eta_min = 1e-5
epochs = 50
patience = 5
log_interval = 10
device = 'cuda'
data_path = '../workspace/frames/frames'

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

begin_frame, end_frame, skip_frame = 0, 4999, 500
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

od_col = []
do1_col = []
hd_col = []
nl_col = []
do2_col = []
bs_col = []
best_loss_col = []
ang_freq = [1, 3, 10, 30, 100]

for i, freq in enumerate(ang_freq):
    for od in output_dim_cnn:
        for do1 in drop_p_cnn:
            for hd in hidden_dim:
                for nl in num_layers:
                    for do2 in drop_p_lstm:
                        for bs in batch_size: 
                            od_col.append(od)
                            do1_col.append(do1)
                            hd_col.append(hd)
                            nl_col.append(nl)
                            do2_col.append(do2)
                            bs_col.append(bs)
                            save_model_path = f'./cnnlstm/single/{freq}/weights/od{od}_do1{do1}_hd{hd}_nl{nl}_do2{do2}_bs{bs}/'
                            save_pred_path = f'./cnnlstm/single/{freq}/predictions/od{od}_do1{do1}_hd{hd}_nl{nl}_do2{do2}_bs{bs}/'
                            log_path = f'./cnnlstm/single/{freq}/log/od{od}_do1{do1}_hd{hd}_nl{nl}_do2{do2}_bs{bs}/'
                            for paths in [log_path, save_model_path, save_pred_path]:
                                os.makedirs(paths, exist_ok=True)

                            # logger
                            logging.basicConfig(level=logging.INFO, filemode='w', format="%(message)s")
                            logger = logging.getLogger()
                            logger.addHandler(logging.FileHandler(log_path + '/log.log'))
                            logger.info(f'cnn output dim: {od}, cnn dropout: {do1}, hidden dim: {hd}, num layers: {nl}, lstm dropout: {do2}, batch size: {bs}')

                            params = {'batch_size': bs, 'shuffle': True, 'num_workers': 20, 'pin_memory': True}
                            test_params = {'batch_size': bs, 'shuffle': False, 'num_workers': 20, 'pin_memory': True}

                            transform = transforms.Compose([transforms.Resize([128, 128]),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.5], std=[0.5])])

                            train_set, valid_set, test_set = CustomDataset(data_path, train_list, train_label[:, i], selected_frames, transform=transform), \
                                                            CustomDataset(data_path, val_list, val_label[:, i], selected_frames, transform=transform), \
                                                            CustomDataset(data_path, test_list, test_label[:, i], selected_frames, transform=transform)
                                                            
                            train_loader = DataLoader(train_set, **params)
                            valid_loader = DataLoader(valid_set, **test_params)
                            test_loader = DataLoader(test_set, **test_params)

                            encoder = EncoderCNN(drop_p=do1, output_dim=od).to(device)
                            decoder = DecoderLSTM(input_dim=od, hidden_dim=hd, num_layers=nl, output_dim=output_dim, drop_p=do2).to(device)

                            criterion = nn.L1Loss()
                            optimizer = torch.optim.AdamW(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate)
                            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

                            trainer = Trainer(train_loader, valid_loader, test_loader, 
                                            [encoder, decoder], optimizer, criterion, 
                                            epochs, scheduler, patience, 
                                            save_model_path, logger, log_interval, device)

                            best_loss = trainer.train()
                            best_loss_col.append(best_loss)
                            
                            logging.shutdown()
                        
                            result = pd.DataFrame({'output cnn': od_col, 
                                                'dropout cnn': do1_col, 
                                                'hidden dim': hd_col, 
                                                'num layers': nl_col, 
                                                'dropout lstm': do2_col, 
                                                'batch size': bs_col, 
                                                'best loss': best_loss_col})
                            result.to_csv(f'finetune_result/cnnlstm_single_{freq}.csv', index=None)
                        
                            true, pred = trainer.predict()
                            true = true*np.array(iqr[i]) + np.array(median[i])
                            pred = pred*np.array(iqr[i]) + np.array(median[i])
                            with open(os.path.join(save_pred_path, f'cnnlstm_single_{freq}.pickle'), 'wb') as f:
                                pickle.dump([true, pred], f)