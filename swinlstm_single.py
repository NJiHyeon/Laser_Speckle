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
from model.SwinLSTM_2 import *
from trainer import Trainer
from utils import seed_everything

seed_everything(42)
# == arguments & parameters == #
# model
embed_dims = [128]
patch_sizes = [8]
window_sizes = [4, 8]
depths = [[8], [12]]
num_heads = [[4]]

# FC
hidden_dim_fc = [512]
# general config
skip_frames = [500, 250]
output_dim=1
# train
batch_sizes = [2, 4]
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


ed_col = []
ps_col = []
ws_col = []
dt_col = [] 
nh_col = [] 
hd_col = [] 
sf_col = []
bs_col = []
best_loss_col = []
ang_freq = [1, 3, 10, 30, 100]


for i, freq in enumerate(ang_freq):
    for ed in embed_dims:
        for ps in patch_sizes: 
            for ws in window_sizes:
                for dt in depths:
                    for nh in num_heads:
                        for hd in hidden_dim_fc:
                            for sf in skip_frames:
                                for bs in batch_sizes:
                                    ed_col.append(str(ed))
                                    ps_col.append(str(ps))
                                    ws_col.append(str(ws))
                                    dt_col.append(str(dt))
                                    nh_col.append(str(nh))
                                    hd_col.append(str(hd))
                                    sf_col.append(str(sf))
                                    bs_col.append(str(bs))
                                    save_model_path = f'./swinlstm/single/{freq}/weights/ed{ed}_ps{ps}_ws{ws}_dt{dt}_nh{nh}_hd{hd}_sf{sf}_bs{bs}/'
                                    save_pred_path = f'./swinlstm/single/{freq}/predictions/ed{ed}_ps{ps}_ws{ws}_dt{dt}_nh{nh}_hd{hd}_sf{sf}_bs{bs}/'
                                    log_path = f'./swinlstm/single/{freq}/log/ed{ed}_ps{ps}_ws{ws}_dt{dt}_nh{nh}_hd{hd}_sf{sf}_bs{bs}/'
                                    for paths in [log_path, save_model_path, save_pred_path]:
                                        os.makedirs(paths, exist_ok=True)

        
                                    print(f'Ang freq: {freq}, Embed dim: {ed}, Patch size:{ps}, Window size:{ws}, depth: {dt}, num head:{nh}, Skip frame:{sf}, Batch size:{bs}')

                                    begin_frame, end_frame, skip_frame = 0, 4999, sf
                                    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()
                                    t_dim = len(selected_frames)

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

                                    model = SwinLSTM(img_size=128, patch_size=ps, in_chans=1, embed_dim=ed, depths=dt,
                                                    num_heads=nh, window_size=ws, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                                                    hidden_dim_fc=hd, dropout_fc=0.0, output_n=1).to(device)

                                    criterion = nn.L1Loss()
                                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                                    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

                                    trainer = Trainer(train_loader, valid_loader, test_loader,
                                                    model, optimizer, criterion, 
                                                    epochs, scheduler, patience, 
                                                    save_model_path, log_interval, device)

                                    best_loss = trainer.train(model_name='SwinLSTM2',depths=dt)
                                    best_loss_col.append(best_loss)
                                    
                                    result = pd.DataFrame({'Embed dim' : ed_col,
                                                        'patch_sizes' : ps_col,
                                                        'window_sizes' : ws_col,
                                                        'depths' : dt_col,
                                                        'num_heads' : nh_col,
                                                        'hidden_dim_fc' : hd_col,
                                                        'skip_frames' : sf_col,
                                                        'batch_sizes' : bs_col,
                                                        'best loss' : best_loss_col})
                                    result.to_csv(f'finetune_result/swinlstm_single_{freq}.csv', index=None)

                                    true, pred = trainer.predict(model_name='SwinLSTM2',depths=dt)
                                    true = true*np.array(iqr[i]) + np.array(median[i])
                                    pred = pred*np.array(iqr[i]) + np.array(median[i])
                                    with open(os.path.join(save_pred_path, f'swinlstm_single_{freq}.pickle'), 'wb') as f:
                                        pickle.dump([true, pred], f)
                                    f.close()