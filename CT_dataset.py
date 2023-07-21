import numpy as np
import random
import glob
import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def sample_random_noise(size):
    return np.random.uniform(low=0.0, high=1.0, size=size)

def get_index(path, slice_number_paths):
    for index, i in enumerate(slice_number_paths):
        if i == path:
            return index
    raise

class CT_Dataset(Dataset):
    def __init__(self, folder, seq_length=23, h=64, w=512, circle_num=1152, lost_interval=2, norm_max=706398, norm_min=0):
        self.data_list = glob.glob(os.path.join(folder, '*', '*'))
        self.seq_length = seq_length
        self.lost_interval = lost_interval
        self.circle_num = circle_num
        self.h = h
        self.w = w
        
        self.T = transforms.ToTensor() # range [0, 255] -> [0.0,1.0]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.max = norm_max
        self.min = norm_min
        self.range = self.max - self.min
        
        self.slice_sort = np.load('../slice_sort.npy')
        self.index_array = np.load('../index_array.npy')
        
    
    def __getitem__(self, index):
        path = self.data_list[index]
        slice_index = get_index(path, self.slice_sort)

        index_array = np.expand_dims(self.index_array[slice_index], axis=0)
        slice_index = float(path.split('-')[-1]) / 360.0
        index_array = np.ones(index_array.shape) * slice_index

        # fetch seq_length data with random start index
        origin_projs = []
        start_index = random.randint(0, self.circle_num-1-self.seq_length)
        for i in range(start_index, start_index+self.seq_length):
            np_path = os.path.join(path, '{}.npy'.format(i))
            proj = np.load(np_path)
            
            # normailze
            proj = (proj - self.min) / self.range
            
            # expand dimension to [height, width, channel]
            proj = np.expand_dims(proj, axis=-1)
            
            origin_projs.append(proj)
            
        origin_projs = np.array(origin_projs)
        
        # augmentation
        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        
        aug_proj = np.zeros((self.seq_length, 1, self.h, self.w))
        for i, p in enumerate(origin_projs):
            aug_proj[i] = self.transform(p).numpy()
        
        # generate lost projection
        lost_proj = aug_proj.copy()
        for i in range(self.seq_length):
            if i % self.lost_interval != 0:
                lost_proj[i] = sample_random_noise((1, self.h, self.w))
                
        # generate index channel
        input_projs = np.zeros((self.seq_length, 4, self.h, self.w))
        for index, i in enumerate(range(start_index, start_index+self.seq_length)):
            ones_array = np.ones([1, self.h, self.w])
            
            # get angle from index
            theta = (2*np.pi / 1152) * i
            cos = np.cos(theta)
            sin = np.sin(theta)
            
            # normalization angle
            cos = (cos - (-1)) / 2
            sin = (sin - (-1)) / 2
            
            cos *= ones_array
            sin *= ones_array 
            
            input_projs[index] = np.concatenate((lost_proj[index], cos, sin, index_array), 0)
            
        return path, input_projs, aug_proj
    
    def __len__(self):
        return len(self.data_list)