"""Define dataset for specific control problem"""

import os
import torch
import load_data
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from constants import *

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate(batch):
    """Set up image data as time series for LSTM model

    Args:
        batch (list): list of tuples of tensors (feature,label)

    Returns:
        tuple: tuple containing updated features and labels
    """
    
    # Reorganise batch dimensions
    
    x,y = list(map(list, zip(*batch)))
    x = torch.stack(x).float()
    y = torch.stack(y).float()/255
    
    x = x.reshape(x.shape[0]//NUM_INDEPENDENT_CLIPS,NUM_INDEPENDENT_CLIPS,*x.shape[1:])
    y = y.reshape(y.shape[0]//NUM_INDEPENDENT_CLIPS,NUM_INDEPENDENT_CLIPS,*y.shape[1:])
    x = x.permute(1,0,2)
    y = y.permute(1,0,2,3,4)
    
    batch = (x,y)
    
    return batch

def val_collate(batch):
    """Set up image data as time series for LSTM model

    Args:
        batch (list): list of tuples of tensors (feature,label)

    Returns:
        tuple: tuple containing updated features and labels
    """
    
    # Reorganise batch dimensions
    
    x,y = list(map(list, zip(*batch)))
    x = torch.stack(x).float()
    y = torch.stack(y).float()/255
    
    x = x.reshape(x.shape[0],1,*x.shape[1:])
    y = y.reshape(y.shape[0],1,*y.shape[1:])
    x = x.permute(1,0,2)
    y = y.permute(1,0,2,3,4)
    
    batch = (x,y)
    
    return batch

class RobotDataset(Dataset):
    def __init__(self, forces_path, img_dir, frames, 
                 img_pref='im', img_type='png', transform=None, target_transform=None):
        self.forces = load_data.load_forces(forces_path, frames)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # To identify image names
        self.img_pref = img_pref
        self.img_type = img_type
    
    def __len__(self):
        return len(self.forces)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_pref + str(idx+1) + '.' + self.img_type)
        image = read_image(img_path)
        force = self.forces[idx]
        if self.transform:
            #force = self.transform(force)
            force = torch.from_numpy(force)
        if self.target_transform:
            image = self.target_transform(image)
        return force, image
    
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
        