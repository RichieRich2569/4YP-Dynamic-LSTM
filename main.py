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
    
    # Load frames
    frames = 36156*4
    val_frames = 3058
    #frames = 300*4
    data = RobotDataset(
        forces_path='main_data/forces.txt',
        img_dir='main_data',
        frames=frames,
        transform=ToTensor()
    )
    val_data = RobotDataset(
        forces_path='validation_data/forces.txt',
        img_dir='validation_data',
        frames=val_frames,
        transform=ToTensor()
    )
    
    dataloader = MultiEpochsDataLoader(data, batch_size=NUM_FRAMES_PER_BATCH*NUM_INDEPENDENT_CLIPS, collate_fn=collate)
    val_loader = MultiEpochsDataLoader(val_data, batch_size=NUM_FRAMES_PER_BATCH, collate_fn=val_collate)
    
    # X = {}
    # Y = {}
    # for i, (x,y) in enumerate(val_loader):
    #    if i > 2:
    #         break
    #    X[i] = x
    #    Y[i] = y
    # x1 = X[0].to(device)
    # y1 = Y[0].to(device)
    # x2 = X[1].to(device)
    # y2 = Y[1].to(device)
    # x3 = X[2].to(device)
    # y3 = Y[2].to(device)
    
    #lstm_model,optim,criterion=model.create_model_functions()
    #lstm_model = model.train_model(lstm_model,optim,criterion,dataloader,num_epochs=30000,save_into_path='checkpoint.pth',val_loader=val_loader)
    #model.save_model(lstm_model, 'dynamic-model.pth')
    
    lstm_model,optim,criterion = model.load_model('dynamic-model.pth')
    lstm_model.train(False)
    
    #lstm_model = model.train_model(lstm_model,optim,criterion,dataloader,num_epochs=8000)
    #model.save_model(lstm_model, 'trained-model.pth')
    # Play video of output
    # y_des = Image.open('test_data/im-paint3.png').convert('L')
    # y_des = torch.from_numpy(np.array(y_des)/255).to(device)
    # print(y_des.shape)
    # param = optimiser.optimise_inputs(lstm_model, y_des, 2)
    
    # N = int(SAMPLING_FREQ*2) # number of frames to be generated
    # X = np.linspace(np.zeros(NUM_CONTROL_INPUTS),param,N)
    # #X = np.reshape(param, (N,6))
    # x = np.concatenate((np.zeros((100,NUM_CONTROL_INPUTS)), X))
    
    # #Reshape data to network input shape and convert to tensor
    # x = torch.from_numpy(x).type(torch.float32).to(device)
    # x = x.unsqueeze(0)
    # x1 = torch.zeros(1,100,6)
    # x2 = 3*torch.rand(1,100,6)
    # out = model.get_output(lstm_model, torch.cat([x1,x2],axis=1))
    
    #out = out - 0.7*torch.mean(out,1)
    #out = (out < 0.15).type(torch.float32)
    
    #y1 = y1 - 0.7*torch.mean(y1,1)
    #y1 = (y1 < 0.15).type(torch.float32)
    
    out = model.get_output(lstm_model, torch.cat([x for i,(x,y) in enumerate(val_loader)],axis=1))
    y = torch.cat([y for i,(x,y) in enumerate(val_loader)],axis=1)
    
    #util.save_video(out1[0,...].detach(), 'prediction-sim-1000.avi')
    # util.save_video(out[0,400:500,...], 'prediction-100.avi')
    # util.save_video(y2[0,100:200,...], 'real-100.avi')
    # util.play_video(y1[0,...].detach())
    util.save_video(out[0,...].detach(), 'validation-prediction.avi')
    util.save_video(y[0,...].detach(), 'validation-real.avi')
    # util.play_video(y[0,...].detach())
    # util.play_video(out[0,...].detach())
    #util.display_image(out[:,-1,...])
    
    # Verify data is of the correct type
    #error = util.check_valid_samples(img,f)
    
    #model.prepare_data(f,img)
    
    return 0
    
# Call main
if __name__ == "__main__":
    sys.exit(main())    
# end main