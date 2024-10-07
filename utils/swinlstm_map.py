import cv2
import os
import random
import pickle
import logging
import numpy as np
import pandas as pd
import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce

from dataset import *
from model.SwinLSTM_2 import *
from trainer import Trainer
from utils import seed_everything

def model_forward_single_layer(model, inputs, num_layers):
    outputs = []
    states = [None] * len(num_layers)
    inputs_len = inputs.shape[1]

    for i in range(inputs_len - 1):
        states, output, attn = model(inputs[:, i], states)
        outputs.append(output)
    return output, attn


seed_everything(42)
# == arguments & parameters == #
# model
embed_dim = 128
patch_size = 8
window_size = 8
depths = [12, 8, 12, 12, 12]
num_head = [4]
# fc
hidden_dim_fc = 512
# train
batch_size = 2
learning_rate = 1e-4
T_max = 30
eta_min = 1e-5
patience = 5
log_interval = 10
device = 'cuda'
data_path = '../workspace/frames/frames'
# etc
ang_freq = [1, 3, 10, 30, 100]
begin_frame, end_frame, skip_frame = 0, 4999, 500
selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

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

val_list = valid_list_label[0]
val_label = np.array(valid_list_label[1])


for i, freq in enumerate(ang_freq):
    depth = [depths[i]]
    test_params = {'batch_size': 1, 'shuffle': False, 'num_workers': 20, 'pin_memory': True}
    transform = transforms.Compose([transforms.Resize([128, 128]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
    valid_set = CustomDataset(data_path, val_list, val_label[:, i], selected_frames, transform=transform)
    valid_loader = DataLoader(valid_set, **test_params)


    model = SwinLSTM(img_size=128, patch_size=patch_size, in_chans=1, embed_dim=embed_dim, depths=depth,
                    num_heads=num_head, window_size=window_size, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                    hidden_dim_fc=hidden_dim_fc, dropout_fc=0.0, output_n=1).to(device)
    
    folder_path = f'./swinlstm/single/{freq}/weights/ed{embed_dim}_ps{patch_size}_ws{window_size}_dt{depth}_nh{num_head}_hd{hidden_dim_fc}_sf{skip_frame}_bs{batch_size}'
    file_name = os.listdir(folder_path)[0]
    model_path = os.path.join(folder_path, file_name)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        for X, y in valid_loader:
            X, y = X.to(device), y.to(device)
            _, attentions = model_forward_single_layer(model, X, depth) #attention 출력 추가 [12, batch(2), 16, 16, 128]
            att_mat = torch.stack(attentions).squeeze(1)

            att_mat = att_mat.cpu().detach()

            att_mat = reduce(att_mat, 'b len1 len2 h -> b len1 len2', 'mean')

            residual_att = torch.eye(att_mat.size(1))
            
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

            v = joint_attentions[-1] #[16, 16]
            grid_size = int(np.sqrt(aug_att_mat.size(-1))) #4
            mask = v[0, :].reshape(grid_size, grid_size).detach().numpy() #[0, 1:]
            mask = cv2.resize(mask / mask.max(), (X.shape[-1], X.shape[-2]))[..., np.newaxis]
            X1 = X[:, -1][0]
            result = (mask.squeeze() * X1.cpu().numpy()) #.astype("unit8")

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 12))

            ax1.set_title('Original')
            ax2.set_title('Attention Mask')
            ax3.set_title('Attention Map')

            _ = ax1.imshow(np.transpose(X1.cpu().numpy(), (1,2,0)))
            _ = ax2.imshow(mask)
            _ = ax3.imshow(np.transpose(result, (1,2,0)))
            plt.savefig(f'./attn_map/freq{freq}_i{i}.png')
