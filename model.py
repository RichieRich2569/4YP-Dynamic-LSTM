"""Contains particular functions/procedures for C-LSTM handling"""

import numpy as np
import torch.nn as nn
import torch
import util
import time
import threading as th
from torch.optim import Adam
from ConvLSTM import Model
from torch.utils.data import DataLoader

from constants import *


# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training can be stopped and saved any time
stop_loop = False
def key_capture_thread():
    global stop_loop
    input()
    stop_loop = True

def create_model_functions():
    """Creates model, optimiser and loss functions"""
    # model = Model(kernel_size=3, cnn_dim=[64,64,64,128,256,256,128,64,64,1],
    #              clstm_hidden_dim=[1,2,2,1])
    model = Model(device, NUM_CONTROL_INPUTS, [16, 32, 32], [32, 32, 64], [64, 128, 128], 8)
    model = model.to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    # MSE Loss - regression
    criterion = nn.MSELoss(reduction='sum')
    
    return model, optim, criterion

def train_model(model,optim,criterion,train_loader,num_epochs=60,val_loader=None,save_into_path=None,load_from_path=None):
    loss_values = []
    epochs = []
    val_loss_values = []
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    epoch = 1
    first_epoch = 0
    
    if load_from_path is not None:
        # Load training parameters
        checkpoint = torch.load(load_from_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['epochs']
        loss_values = checkpoint['loss_values']
        val_loss_values = checkpoint['val_loss_values']
        first_epoch = epochs[-1]
        epoch = first_epoch+1
        
        
    torch.backends.cudnn.benchmark = True
    while not stop_loop and epoch-first_epoch <= num_epochs:
        if epoch % DISPLAY_LOSS_EVERY == 0 or epoch in [1,2,3,4,5]:          
            train_loss = 0
        model.train()
        optim.zero_grad()
        
        # Initial hidden state for ConvLSTM - Update if ConvLSTM changed
        hidden = model.init_states(N=NUM_INDEPENDENT_CLIPS)
        hidden_states = [(None,hidden)]
        for i, (force, img) in enumerate(train_loader):
            force = force.to(device)
            img = img.to(device)
            
            hidden = detach_and_grad(hidden_states[-1][1])
                                                                                              
            output, new_hidden = model(force, hidden, return_hidden=True)
            hidden_states.append((hidden,new_hidden))
            
            while len(hidden_states) > BPTT_K2:
                del hidden_states[0]
                #del forces[0]
            
            if (i+1)%BPTT_K1 == 0:
                loss = criterion(output,img) + model.sampling_loss
                loss.backward(retain_graph=(BPTT_K2 > BPTT_K1))
                
                if epoch % DISPLAY_LOSS_EVERY == 0 or epoch in [1,2,3,4,5]:
                    train_loss += loss.item()
                
                for j in range(BPTT_K2-1):
                    if hidden_states[-j-2][0] is None:
                        break
                    for k in range(3):
                        for l in range(2):
                            curr_grad = hidden_states[-j-1][0][k][l].grad
                            hidden_states[-j-2][1][k][l].backward(curr_grad, retain_graph=(BPTT_K1<BPTT_K2))     
        optim.step()
        
        if epoch%DISPLAY_LOSS_EVERY == 0 or epoch in [1,2,3,4,5]:
            train_loss /= len(train_loader)*torch.numel(output)                    
            loss_values.append(train_loss)
            epochs.append(epoch)
            
            # Perform validation check
            if VALIDATION:
                val_loss = validate(model,optim,criterion,val_loader)
                val_loss_values.append(val_loss)
                print("Epoch:{} Training Avg Loss:{:.5f} Validation Avg Loss:{:.5f}\n".format(
                      epoch, train_loss, val_loss))
                if val_loss <= VALIDATION_ERROR:
                    # If validation error low enough, end training
                    break
            else:
                print("Epoch:{} Training Avg Loss:{:.5f}\n".format(
                      epoch, train_loss))
        epoch+=1

    if VALIDATION:
        util.plot_loss(val_loss_values, epochs)
    if save_into_path is not None:
        # Save training parameters
        torch.save({
            'epochs':epochs,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'loss_values':loss_values,
            'val_loss_values':val_loss_values
        }, save_into_path)
        
    util.plot_loss(loss_values, epochs)
    return model

def validate(model,optim,criterion,val_loader):
    """Perform validation pass, return error"""
    model.eval()
    # Initial hidden state for model
    hidden = model.init_states(N=1)
    hidden_states = [(None,hidden)]
    
    val_loss = 0
    
    # Perform validation pass
    for i, (force, img) in enumerate(val_loader):
        force = force.to(device)
        img = img.to(device)
        
        hidden = detach_and_grad(hidden_states[-1][1])
                                                                                            
        output, new_hidden = model(force, hidden, return_hidden=True)
        hidden_states.append((hidden,new_hidden))
        
        while len(hidden_states) > BPTT_K2:
            del hidden_states[0]
        
        loss = criterion(output,img) + model.sampling_loss    
        val_loss += loss.item()
        
    return val_loss / (len(val_loader)*torch.numel(output))
    
def detach_and_grad(state):
    new_state = []
    for i in range(len(state)):
        h = state[i][0].detach()
        c = state[i][1].detach()
        h.requires_grad = True
        c.requires_grad = True
        new_state.append((h,c))
    return new_state

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(path):
    model, optim,criterion = create_model_functions()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model, optim, criterion

def get_output(model, x):
    out_list = []
    T = x.shape[1]
    N = x.shape[0]
    h = model.init_states(N=N)
    for i in range(0,T//500):
        xi = x[:,i*500:(i+1)*500,...]
        out, h = model(xi.to(device), h, return_hidden=True)
        out_list.append(out.detach())
    if (T % 500) != 0:
        xi = x[:,500*(T//500):T,...]
        out, h = model(xi.to(device), h, return_hidden=True)
        out_list.append(out.detach())
    return torch.cat(out_list, dim=1)