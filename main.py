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
    f = 2 # in Hz
    t = np.linspace(0,10,10*30)
    xsin = np.zeros((len(t),6))
    xsin[:,0] = 0.5*(np.sin(t*2*np.pi*2)+1.5)
    xsin[:,1] = 0.5*(np.sin(t*2*np.pi*1)+2)
    xsin[:,2] = 0.5*(np.sin(t*2*np.pi*0.5)+5)
    xsin[:,3] = 0.5*(np.sin(t*2*np.pi*0.7)+2)
    xsin[:,4] = 0.5*(np.sin(t*2*np.pi*1.5)+3.2)
    xsin[:,5] = 0.5*(np.sin(t*2*np.pi*1)+1)
    x = torch.cat((torch.zeros(1,120,6),torch.Tensor(xsin).unsqueeze(0)),axis=1).to(device)
    out = model.get_output(lstm_model, x)
    
    # Obtain output via optimisation
    
    # Open image
    # y1 = Image.open('optimisation_img/im2.png').convert('L')
    # y1 = torch.from_numpy(np.array(y1)/255).to(device)
    # y2 = Image.open('optimisation_img/im1.png').convert('L')
    # y2 = torch.from_numpy(np.array(y2)/255).to(device)
    
    # # Parameter definition and optimisation
    # T = 5 # 2 Seconds maximum time
    # t = [1,2,4,5]
    # y_des = [y1,y1,y2,y2]
    # param = optimiser.optimise_inputs(lstm_model, y_des, t, T)
    
    # # Obtain X from optimised parameters, show output
    # N = int(SAMPLING_FREQ*T)
    # n = (SAMPLING_FREQ*torch.tensor(t)).type(torch.int)
    # x = np.reshape(param, (len(n),6))
    # X = optimiser.piecewise_generator(x, n, N)
    # out = model.get_output(lstm_model, X)
    
    # Play video from prediction
    #util.play_video(out[0,...])
    util.save_video(out[0,...], 'control_data/videos/sin4.avi')
    util.save_data(x, 'control_data/sin4.txt')
    
    return 0
    
# Call main
if __name__ == "__main__":
    sys.exit(main())    
# end main