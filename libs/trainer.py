import os 
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import ModelCheckpoint

def model_forward_single_layer(model, inputs, num_layers):
    outputs = []
    states = [None] * len(num_layers)
    inputs_len = inputs.shape[1]

    for i in range(inputs_len - 1):
        states, output, attn = model(inputs[:, i], states)
        outputs.append(output)
    return output, attn

class Trainer():
    def __init__(self, train_loader, valid_loader, test_loader, 
                 model, optimizer, criterion, 
                 epochs, scheduler, patience, 
                 best_model_path, log_interval, device):
        # data
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        # model
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = scheduler
        self.patience = patience
        self.best_model_path = best_model_path
        # supplementary
        self.log_interval = log_interval
        self.device = device
        if isinstance(model, list):
            self.encoder_ckpt = ModelCheckpoint(best_model_path, 1)
            self.decoder_ckpt = ModelCheckpoint(best_model_path, 1)
        else:
            self.model_ckpt = ModelCheckpoint(best_model_path, 1)
            
    def train(self, model_name=None, depths=None):        
        
        best_loss = np.inf
        patience_count = 0
        for epoch in range(1, self.epochs+1):
            if isinstance(self.model, list):
                # CNN-LSTM, CNN-GRU
                encoder, decoder = self.model
                encoder.train()
                decoder.train()
            else:
                # ConvLSTM, 3DCNN, SwinLSTM
                self.model.train()
            
            N_count = 0
            losses = []
            for batch_idx, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                N_count += X.size(0)

                self.optimizer.zero_grad()
                if model_name == 'SwinLSTM':
                    states = [None]
                    inputs_len = len(X)
                    for i in range(inputs_len):
                        output, states = self.model(X[:, i], states)
                elif model_name == 'CNNLSTM':
                    output = decoder(encoder(X))
                elif model_name == 'SwinLSTM2':
                    output, attention = model_forward_single_layer(self.model, X, depths)
                else:
                    output = self.model(X)
                
                loss = self.criterion(output.squeeze(), y.squeeze())
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
 
            loss_val = self.valid_step(model_name=model_name, depths=depths)
            print(f'Epoch [{epoch}/{self.epochs}] Train Loss {np.mean(losses):.6f} Validation Loss {loss_val:.6f}')
            
            if loss_val <= best_loss:
                best_loss = loss_val
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    break

            if isinstance(self.model, list):
                self.encoder_ckpt.save_checkpoint(encoder.state_dict(), 'encoder', epoch, loss_val)
                self.decoder_ckpt.save_checkpoint(decoder.state_dict(), 'decoder', epoch, loss_val)
            else:
                self.model_ckpt.save_checkpoint(self.model.state_dict(), 'model', epoch, loss_val)
                
        return best_loss
                
    def valid_step(self, model_name=None, depths=None):
        if isinstance(self.model, list):
            encoder, decoder = self.model
            encoder.eval()
            decoder.eval()
        else:
            self.model.eval()

        losses = []
        with torch.no_grad():
            for X, y in self.valid_loader:
                X, y = X.to(self.device), y.to(self.device)

                if model_name == 'SwinLSTM':
                    states = [None]
                    inputs_len = len(X)
                    for i in range(inputs_len):
                        output, states = self.model(X[:, i], states)
                elif model_name == 'CNNLSTM':
                    output = decoder(encoder(X))
                elif model_name == 'SwinLSTM2':
                    output, attention = model_forward_single_layer(self.model, X, depths)
                else:
                    output = self.model(X)
                    
                loss = self.criterion(output.squeeze(), y.squeeze())
                losses.append(loss.item())
        
        return np.mean(losses)

    def predict(self, model_name, depths=None): 
        true = []
        pred = []
        if isinstance(self.model, list):
            encoder, decoder = self.model
            ckpt_list = os.listdir(self.best_model_path)
            encoder_ckpt = [i for i in ckpt_list if 'encoder' in i][0]
            decoder_ckpt = [i for i in ckpt_list if 'decoder' in i][0]
            encoder.load_state_dict(torch.load(os.path.join(self.best_model_path, encoder_ckpt)))
            decoder.load_state_dict(torch.load(os.path.join(self.best_model_path, decoder_ckpt)))
            encoder.eval()
            decoder.eval()
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.best_model_path, os.listdir(self.best_model_path)[0])))
            self.model.eval()
            
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                if model_name == 'SwinLSTM':
                    states = [None]
                    inputs_len = len(X)
                    for i in range(inputs_len):
                        output, states = self.model(X[:, i], states)
                elif model_name == 'CNNLSTM':
                    output = decoder(encoder(X))
                elif model_name == 'SwinLSTM2':
                    output, attention = model_forward_single_layer(self.model,X,depths)
                else:
                    output = self.model(X)
                
                pred.extend(output.cpu().data.squeeze().numpy())
                true.extend(y.cpu().data.squeeze().numpy())
        
        return true, pred