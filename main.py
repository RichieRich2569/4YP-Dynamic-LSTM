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
    val_x =  torch.from_numpy(ld.load_forces('input_data/validation.txt', 3058)).type(torch.float32).to(device)
    val_x = val_x.reshape(val_x.shape[0],1,*val_x.shape[1:]).permute(1,0,2)
    train_x = torch.from_numpy(ld.load_forces('input_data/training.txt', 36156*4)).type(torch.float32).to(device)
    train_x = train_x.reshape(train_x.shape[0]//NUM_INDEPENDENT_CLIPS,NUM_INDEPENDENT_CLIPS,*train_x.shape[1:]).permute(1,0,2)
    
    # Load trained model for testing
    lstm_model,optim,criterion = model.load_model('dynamic-model.pth')
    lstm_model.train(False)
    
    # Obtain output via preset input

    # Create sinusoid - f
    # f = 0.2 # in Hz
    # t = np.linspace(0,10,30*30)
    # xsin = np.zeros((len(t),6))
    # xsin[:,0] = 2*np.random.rand(1)*(np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+3)
    # xsin[:,1] = np.random.rand(1)*3
    # xsin[:,2] = 2*np.random.rand(1)*(np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+5)
    # xsin[:,3] = np.random.rand(1)*3
    # xsin[:,4] = 2*np.random.rand(1)*(np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+3.2)
    # xsin[:,5] = np.random.rand(1)*3
    
    # x = torch.cat((torch.zeros(1,120,6),torch.Tensor(xsin).unsqueeze(0)),axis=1).to(device)
    # out = model.get_output(lstm_model, x)
    
    # Obtain output via optimisation
    
    # Open image
    y1 = Image.open('optimisation_img/im1.png').convert('L')
    y1 = torch.from_numpy(np.array(y1)/255).to(device)
    # Parameter definition and optimisation
    T = 3 # 2 Seconds maximum time
    t = [2]
    y_des = [y1]
    param = optimiser.optimise_inputs(lstm_model, y_des, t, T)
    
    # Obtain X from optimised parameters, show output
    N = int(SAMPLING_FREQ*T)
    n = (SAMPLING_FREQ*torch.tensor(t)).type(torch.int)
    x = np.reshape(param, (len(n),6))
    X = optimiser.piecewise_generator(x, n, N)
    out = model.get_output(lstm_model, X)
    
    # Play video from prediction
    #util.play_video(out[0,...])
    util.save_video(out[0,...], 'control_data/videos/static_control1.avi')
    util.save_data(X, 'control_data/static_control1.txt')
    
    return 0
    
# Call main
if __name__ == "__main__":
    sys.exit(main())    
# end main