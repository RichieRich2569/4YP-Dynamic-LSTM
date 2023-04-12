"""Load image and forces data from 'training_data' directory"""

# Imports
import numpy as np

from constants import *
from matplotlib import image

def load_img(path, frames, img_pref='im', img_type='png'):
    """Load all image files from given path

    Args:
        path (str): path to folder containing pictures.
        frames (int): number of total image frames considered.
        img_pref (str, optional): image name prefix, leads to files named img_pref + int
                                  i.e. im1, im2, etc. Defaults to 'im'.
        img_type (str, optional): image type, i.e. 'png', 'jpg'. Defaults to 'png'.

    Returns:
        numpy.array: array containing all sampled images
        list: list containing all data id strings - picture names
    """
    # Read all images and store in Numpy array 'img'
    im = []
    for i in range(1,frames+1):
        id = img_pref+str(i)
        p = image.imread(path+'/'+id+'.'+img_type)
        im.append(np.array(p))
    img = np.array(im)
    
    return img

def load_forces(path, frames):
    """Load all force data from text file in given path into images

    Args:
        path (str): Path of text file to load from
        frames (int): Number of frames to load

    Returns:
        numpy.array: array containing all sampled control forces 
    """
    
    f = np.loadtxt(path, delimiter=',')
    
    # Only return number of frames given
    if len(f) >= frames:
        f = f[:frames]
        
    # Obtain input data
    new_f = []
    for i in range(0,len(f)):
        new_f.append(f[i])
    f = np.array(new_f)
    
    return f