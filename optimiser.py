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

def sum_squared_error(Y,y_des,n):
    """Calculate squared error between video and sequence of images at given time points

    Args:
        Y (torch.tensor): tensor corresponding to images in time series (video), format [1,T,C,H,W] 
        y_des (torch.tensor): tensor corresponding to sequential image format [N,C,H,W]
        n (torch.tensor): tensor corresponding to indexes for each y_des in Y

    Returns:
        float: Returns single scalar value corresponding to the summed squared error
    """
    error = 0
    for i, ni in enumerate(n):
        error += float(torch.sum((Y[0,ni,...]-y_des[i])**2))
    
    return error

def piecewise_generator(x, n, N, concatenate_zeros=True):
    """Generate prediction with inputs x at the times given by t, linearly interpolating between each xi

    Args:
        x (np.array): array corresponding to inputs to network
        t (np.array): time points at which each xi map to (ensure all t<=T)
        T (Float): maximum time for prediction
    """
    
    # Obtain piecewise linear data given constraints
    X = np.zeros((N+1,6))
    x0 = np.zeros(6)
    for i, ni in enumerate(n):
        if i == 0:
            X[0:ni+1,:] = np.linspace(x0, x[i,:], ni+1)
        else:
            X[n[i-1]:ni+1,:] = np.linspace(x0, x[i,:], ni+1-n[i-1])
        x0 = X[ni,:]
    if ni < N:
        # Steady state at the end
        X[ni:N+1,:] = np.linspace(x0, x0, N+1-ni)
            
    # If concatenate zeros, append this to start of sequence
    if concatenate_zeros:
        # Add 100 time points at the start with zero value
        X = np.concatenate((np.zeros((100,NUM_CONTROL_INPUTS)), X))
    
    # Reshape data to network input shape and convert to tensor
    X = torch.from_numpy(X).type(torch.float32).to(device)
    X = X.unsqueeze(0)
    
    return X

def calculate_error(x, y_des, error, model, t, T, generator=piecewise_generator):
    """Take parameters from optimiser and return error between prediction and desired output

    Args:
        x (np.array): array corresponding to all parameters
        error (function): function defining error calculation
        y_des (torch.Tensor): Tensor of desired images [N,C,H,W]
        model (C-LSTM Model): Network model for optimisation
        t (List/np.array): Values of t corresponding to images y_des
        T (Float): Maximum time for optimisation
        generator (function): Generator function for interpolating inputs

    Returns:
        float: Returns single scalar value corresponding to error
    """
    # Find positional elements for each time point - maximum and intermediate
    N = int(SAMPLING_FREQ*T)
    n = (SAMPLING_FREQ*torch.tensor(t)).type(torch.int)
    
    # Reshape x - optimiser decouple parameters
    x = np.reshape(x, (len(n),6))
    
    # Get interpolated data
    X = generator(x, n, N)
    
    y = get_output(model, X)
    
    # Calculate error, return
    n = n + 100 # Add to n, as input has been concatenated with zeros
    e = error(y, y_des, n)
    return e
    

def optimise_inputs(network_model, y, t, T, error=sum_squared_error):
    """Optimise inputs in LSTM model to match output

    Args:
        f (Model obj): trained model for LSTM.
        y (torch.tensor): tensor corresponding to image(s) to be matched in optimisation.
        t (List): list value time points matching required images.
        T (Float): maximum time for prediction.

    Returns:
        torch.array: Returns sequence of inputs that best reach the desired shape
    """
    
    bounds = Bounds(np.zeros(len(t)*NUM_CONTROL_INPUTS),MAX_INPUT*np.ones(len(t)*NUM_CONTROL_INPUTS))
    
    minimizer_kwargs = {'options':{'disp':True}}
    res = dual_annealing(calculate_error, bounds, args=(y, error, network_model, t, T), maxiter=100, minimizer_kwargs=minimizer_kwargs)
    
    x_opt = res.x
    
    return x_opt