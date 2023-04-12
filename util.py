"""Module containing useful definitions"""

# Import
from constants import *

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from PIL import Image,ImageTk

def check_valid_samples(img, f):
    """Check whether given 

    Args:
        img (np.array): array containing all samples of training/test images.
        f (np.array): array containing control forces corresponding to each sample.

    Raises:
        ValueError: Image dimensions not square.
        ValueError: Image dimensions not equal to the assigned value.
        ValueError: Sample sizes in image and forces do not match.
        ValueError: Force vector dimension is not assigned value.

    Returns:
        int: Returns value of error. Returns 0 if no error present.
    """
    k,m,n = img.shape
    o,p = f.shape
    
    error = 0
    # Image sizes should be square and consistent
    if m != n:
        error = 1
        raise ValueError("Image dimensions not square, currently {}x{}".format(m,n))
    elif m != IMG_DIM:
        error = 2
        raise ValueError("Image dimensions not equal to the assigned value, currently {}".format(m))
    
    # Sample length should match for both image and forces
    if k != o:
        error = 3
        raise ValueError("Sample sizes in image and forces do not match. Currently {} and {}.".format(k,o))
    
    # Value of p should be 6 - the number of control inputs
    if p != NUM_CONTROL_INPUTS:
        error = 4
        raise ValueError("Force vector dimension is not {}. Currently {}.".format(NUM_CONTROL_INPUTS,p))
    
    return error

def play_video(img):
    """Play video

    Args:
        img (array/list): List/array of sequential images
    """
    
    if isinstance(img,torch.Tensor):
        # Convert tensor (in LSTM form) into numpy array
        img = img.squeeze(1).cpu().numpy()
    
    frames = [] # for storing the generated images
    fig = plt.figure()
    for i in range(len(img)):
        frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=20, blit=True,
                                    repeat_delay=200)
    plt.show()

def save_video(img_list, name='prediction.avi'):
    """Save video

    Args:
        img_list (array/list): List/array of sequential images
    """
    
    if isinstance(img_list,torch.Tensor):
        # Convert tensor (in LSTM form) into numpy array
        img_list = 255*img_list.permute([0,2,3,1]).cpu().numpy()
    
    h, w, c = img_list.shape[1:]
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (720,720))
    
    
    for i in range(len(img_list)):
        img = cv2.resize(img_list[i,...],(720,720),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.write(img.astype('uint8'))
    out.release()

def display_image(img):
    """Display given image.

    Args:
        img (array/list): array representing image
    """
    
    if isinstance(img,torch.Tensor):
        # Convert tensor (in LSTM form) into numpy array
        img = img.squeeze(1).squeeze(0).cpu().numpy()
    
    plt.imshow(img, cmap=cm.Greys_r)

    plt.show()

def plot_loss(values, epochs=None):
    """Plot loss values
    """
    if epochs is None:
        epochs = np.arange(1,len(values)+1)
    plt.plot(epochs, values, c='b')
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    
def save_loss(values, path='loss-1000.png'):
    """Plot loss values
    """

    epochs = np.arange(1,len(values)+1)
    plt.plot(epochs, values, c='b')
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(path)

def save_data(x, file_name):
    """Saves array data in x to text file"""
    # Expecting x to come in default form for input parameters - change format before saving
    x = x.detach()
    if isinstance(x,torch.Tensor):
        # Convert tensor (in LSTM form) into numpy array
        x = x.squeeze(1).cpu().numpy()
    
    np.savetxt(file_name, x)

def load_data(file_name):
    """Loads array data from text file"""
    x = np.loadtxt(file_name)
    x = torch.from_numpy(x).type(torch.float32)
    x = x.unsqueeze(0)
    
    return x
    