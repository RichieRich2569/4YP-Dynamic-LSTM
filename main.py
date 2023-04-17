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
    # f = 10 # in Hz
    # t = np.linspace(0,10,30*30)
    # xsin = np.zeros((len(t),6))
    # xsin[:,0] = (np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+1)
    # xsin[:,1] = (np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+1)
    # xsin[:,2] = (np.sin(t*2*np.pi*f+np.random.rand(1)*np.pi)+1)
    # xsin[:,3] = np.random.rand(1)*3
    # xsin[:,4] = np.random.rand(1)*3
    # xsin[:,5] = np.random.rand(1)*3
    
    # x = torch.cat((torch.zeros(1,120,6),torch.Tensor(xsin).unsqueeze(0)),axis=1).to(device)
    # out = model.get_output(lstm_model, x)
    
    # Obtain output via optimisation
    
    # Open obstacle
    # obs = Image.open('optimisation_img/obstacle3.png').convert('L')
    # obs = torch.from_numpy(np.array(obs)/255).to(device)
    # obs[obs>=0.1] = 1
    # obs[obs<0.1] = 0
    # obs = 1 - obs
    obs = None
    
    
    # Open image
    y1 = Image.open('optimisation_img/tip1.png').convert('L')
    y1 = torch.from_numpy(np.array(y1)/255).to(device)
    y2 = Image.open('optimisation_img/tip2.png').convert('L')
    y2 = torch.from_numpy(np.array(y2)/255).to(device)
    y3 = Image.open('optimisation_img/tip3.png').convert('L')
    y3 = torch.from_numpy(np.array(y3)/255).to(device)
    y4 = Image.open('optimisation_img/tip4.png').convert('L')
    y4 = torch.from_numpy(np.array(y4)/255).to(device)
    y5 = Image.open('optimisation_img/tip5.png').convert('L')
    y5 = torch.from_numpy(np.array(y5)/255).to(device)
    # Parameter definition and optimisation
    T = 5 # maximum time
    t = [1.0,1.5,2.0,2.5,3.0]
    y_des = [y1,y2,y3,y4,y5]
    param = optimiser.optimise_inputs(lstm_model, y_des, t, T, obstacle=obs)

    # Obtain X from optimised parameters, show output
    N = int(SAMPLING_FREQ*T)
    n = (SAMPLING_FREQ*torch.tensor(t)).type(torch.int)
    x = np.reshape(param, (len(n),6))
    X = optimiser.piecewise_generator(x, n, N)
    out = model.get_output(lstm_model, X)
    
    # Play video from prediction
    #out[:,:,:,obs==1]=0
    #util.play_video(out[0,...])
    util.save_video(out[0,...], 'control_data/videos/paint4.avi')
    util.save_data(X, 'control_data/paint4.txt')
    
    return 0
    
# Call main
if __name__ == "__main__":
    sys.exit(main())    
# end main