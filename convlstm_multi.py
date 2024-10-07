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
from model.ConvLSTM import *
from trainer import Trainer
from utils import seed_everything

seed_everything(42)

# == arguments & parameters == #
input_dim = 1
hidden_dim = [16]
kernel_size = [(3,3)]
num_layers = [1, 3]
drop_p = [0.0]
output_dim = 5
batch_size = [4]
batch_norm = [False, True]
skip_frames = [500, 250, 100]
learning_rate = 1e-4
T_max = 30
eta_min = 1e-5
epochs = 50
patience = 5
log_interval = 10
device = 'cuda'
data_path = '../workspace/frames/frames'
finetune = True # If finetune, checkpoint will not be saved

# ============================ #
   
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
nl_col = []
do_col = []
bs_col = []
bn_col = []
sf_col = []
best_loss_col = []


for hd, ks, nl in zip(hidden_dim, kernel_size, num_layers):
    for do in drop_p:
        for bs in batch_size:
            for bn in batch_norm:
                for sf in skip_frames:
                    hd_col.append(str(hd))
                    ks_col.append(str(ks))
                    nl_col.append(str(nl))
                    do_col.append(str(do))
                    bs_col.append(str(bs))
                    bn_col.append(str(bn))
                    sf_col.append(str(sf))
                    save_model_path = f'./convlstm/multi/weights/hd{hd}_do{do}_bs{bs}_bn{bn}_sf{sf}/'
                    save_pred_path = f'./convlstm/multi/predictions/hd{hd}_do{do}_bs{bs}_bn{bn}_sf{sf}/'
                    log_path = f'./convlstm/multi/log/hd{hd}_do{do}_bs{bs}_bn{bn}_sf{sf}/'
                    for paths in [log_path, save_model_path, save_pred_path]:
                        os.makedirs(paths, exist_ok=True)

                    # logger
                    logging.basicConfig(level=logging.INFO, filemode='w', format="%(message)s")
                    logger = logging.getLogger()
                    logger.addHandler(logging.FileHandler(log_path + '/log.log'))
                    logger.info(f'hidden dim: {hd}, kernel size: {ks}, num layers: {nl}, dropout:{do}, batch size: {bs}')

                    begin_frame, end_frame, skip_frame = 0, 4999, sf
                    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

                    params = {'batch_size': bs, 'shuffle': True, 'num_workers': 20, 'pin_memory': True}
                    test_params = {'batch_size': bs, 'shuffle': False, 'num_workers': 20, 'pin_memory': True}

                    transform = transforms.Compose([transforms.Resize([128, 128]),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5], std=[0.5])])

                    train_set, valid_set, test_set = CustomDataset(data_path, train_list, train_label, selected_frames, transform=transform), \
                                                    CustomDataset(data_path, val_list, val_label, selected_frames, transform=transform), \
                                                    CustomDataset(data_path, test_list, test_label, selected_frames, transform=transform)
                                                    
                    train_loader = DataLoader(train_set, **params)
                    valid_loader = DataLoader(valid_set, **test_params)
                    test_loader = DataLoader(test_set, **test_params)

                    model = ConvLSTM(input_dim=input_dim, 
                                    hidden_dim=hd, 
                                    kernel_size=ks, 
                                    num_layers = nl, 
                                    batch_first=True, bias=True, 
                                    drop_p = do, 
                                    output_dim=output_dim, 
                                    batch_norm = bn
                                    ).to(device)

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
                                            'num layers': nl_col, 
                                            'dropout':do_col, 
                                            'batch size': bs_col, 
                                            'batch norm': bn_col, 
                                            'skip frames': sf_col, 
                                            'best loss': best_loss_col})
                    result.to_csv('finetune_result/convlstm_multi.csv', index=None)

                    true, pred = trainer.predict()
                    true = true*iqr + median
                    pred = pred*iqr + median
                    with open(os.path.join(save_pred_path, 'convlstm_multi.pickle'), 'wb') as f:
                        pickle.dump([true, pred], f)
                    f.close()