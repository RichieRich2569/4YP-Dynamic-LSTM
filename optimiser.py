"""Module containing useful definitions"""

# Import
from constants import *

import torch
import numpy as np
from scipy.optimize import minimize, dual_annealing, brute, Bounds
import cv2
from model import get_output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from PIL import Image,ImageTk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def last_squared_error(X,y):
    """Calculate squared error between two images

    Args:
        X (torch.tensor): tensor corresponding to images in time series, format [1,T,C,H,W] 
        y (torch.tensor): tensor corresponding to image format [C,H,W]

    Returns:
        float: Returns single scalar value corresponding to the summed squared error
    """
    x = X[0,-1,...]
    return float(torch.sum((x-y)**2))

def last_ten_squared_error(X,y):
    """Calculate squared error between two images

    Args:
        X (torch.tensor): tensor corresponding to images in time series, format [1,T,C,H,W] 
        y (torch.tensor): tensor corresponding to image format [C,H,W]

    Returns:
        float: Returns single scalar value corresponding to the summed squared error
    """
    x = X[:,-11:-1,...]
    return float(torch.sum((x-y)**2))

def calculate_error(x, y_des, error, model, time):
    """Take parameters from optimiser and return error between prediction and desired output

    Args:
        x (np.array): array corresponding to all parameters
        error (function): function defining error calculation

    Returns:
        float: Returns single scalar value corresponding to error
    """
    
    # Generate data according to ramp increase to x
    N = int(SAMPLING_FREQ*time) # number of frames to be generated
    X = np.linspace(np.zeros(NUM_CONTROL_INPUTS),x,N)
    #X = np.reshape(x, (N,6))
    
    #Concatenate current inputs to a starting vector of zeros to remove initial dynamic effects
    x = np.concatenate((np.zeros((100,NUM_CONTROL_INPUTS)), X))
    
    #Reshape data to network input shape and convert to tensor
    x = torch.from_numpy(x).type(torch.float32).to(device)
    x = x.unsqueeze(0)
    # x = torch.from_numpy(x).type(torch.float32).to(device)
    # x = torch.cat([torch.zeros(1,100,6).to(device),x.repeat(1,N,1)], axis=1)
    
    y = get_output(model, x)
    
    # Calculate error, return
    e = error(y, y_des)
    return e
    

def optimise_inputs(network_model, y, time, error=last_squared_error):
    """Optimise inputs in LSTM model to match output

    Args:
        f (Model obj): trained model for LSTM.
        y (torch.tensor): tensor corresponding to image(s) to be matched in optimisation.
        time (float): value (in seconds) corresponding to amount of time to reach destination shape.

    Returns:
        torch.array: Returns sequence of inputs that best reach the desired shape
    """
    #x0 =MAX_INPUT* np.random.rand(NUM_CONTROL_INPUTS) # Generate first guess of inputs
    #N = int(SAMPLING_FREQ*time)
    
    bounds = Bounds(np.zeros(NUM_CONTROL_INPUTS),MAX_INPUT*np.ones(NUM_CONTROL_INPUTS))
    
    #res = minimize(calculate_error, x0, args=(y, error, network_model, time), bounds=bounds, method='SLSQP', options={'maxiter':10000, 'disp':True})
    #minimizer_kwargs = {'args':(y, error, network_model, time), 'method':"SLSQP", 'bounds':bounds, 'options':{'maxiter':10000, 'disp':True}}
    #res = basinhopping(calculate_error, x0, minimizer_kwargs=minimizer_kwargs)
    minimizer_kwargs = {'options':{'disp':True}}
    res = dual_annealing(calculate_error, bounds, args=(y, error, network_model, time), maxiter=10, minimizer_kwargs=minimizer_kwargs)
    
    x_opt = res.x
    
    return x_opt