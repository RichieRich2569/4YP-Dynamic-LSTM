"""Train, test and save C-LSTM model on robot data - main"""
    
# Imports
import sys
import util
import os
import load_data as ld
import torch
import numpy as np
import optimiser
from PIL import Image
import model
from RobotDataset import RobotDataset, MultiEpochsDataLoader, collate, val_collate
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from constants import *
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def main():
    """Create, train, test and save C-LSTM model"""
    
    # Load input data
    val_x = ld.load_forces('input_data/validation.txt', 3058)
    val_x = val_x.reshape(val_x.shape[0],1,*val_x.shape[1:]).permute(1,0,2)
    train_x = ld.load_forces('input_data/validation.txt', 36156*4)
    train_x = train_x.reshape(train_x.shape[0]//NUM_INDEPENDENT_CLIPS,NUM_INDEPENDENT_CLIPS,*train_x.shape[1:]).permute(1,0,2)
    
    # Load trained model for testing
    lstm_model,optim,criterion = model.load_model('dynamic-model.pth')
    lstm_model.train(False)
    
    # Obtain output from input
    out = model.get_output(lstm_model, val_x)
    
    # Play video from prediction
    util.play_video(out[0,...])
    
    return 0
    
# Call main
if __name__ == "__main__":
    sys.exit(main())    
# end main