import os
import os.path as osp
import math
import random
import pickle
import warnings
import glob

import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.datasets import UCF101
from torchvision.datasets.folder import make_dataset
from tools.video_utils import VideoClips
from torchvision.io import read_video

from utils import set_random_seed
from tools.data_utils import *

import av
from PIL import Image


class Meteoset(Dataset):
    FILE_APPEND = ".png"
    def __init__(self,
                 _path,  # Path to directory or zip.
                 resolution=None,
                 nframes=16,  # number of frames for each video.
                 train=True,
                 interpolate=False,
                 loader=default_loader,  # loader for "sequence" of images
                 return_vid=True,  # True for evaluating FVD
                 cond=False,
                 predictioner=False,
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.root = _path
        self.apply_resize = True

        self.frames_per_sample = nframes
        self.img_resolution = resolution
        self.to_tensor = T.ToTensor()
        self.predictioner = predictioner
        if self.predictioner:
            self.frames_per_sample *= 2

        tiles_dirs = glob.glob(os.path.join(self.root, "*"))
        tiles = {} 
        for tile_dir in tiles_dirs:
            if os.path.isdir(tile_dir):
                tile = os.path.basename(tile_dir)
                tiles[tile] = []
                for file in glob.glob(os.path.join(tile_dir, f"*{self.FILE_APPEND}")):
                    tiles[tile].append(file)
                tiles[tile].sort()
                
        self.sequences = []
        diference = 600
        for tile in tiles:
            for i, file in enumerate(tiles[tile]):
                sequence = []
                file_timestamp = int(os.path.basename(file).split(".")[0])
                for j, file2 in enumerate(tiles[tile][i+1:]):
                    file2_timestamp = int(os.path.basename(file2).split(".")[0])
                    if file2_timestamp == file_timestamp+(j+1)*diference:
                        sequence.append(file2)
                    else:
                        break
                    if len(sequence) == self.frames_per_sample:
                        break
                if len(sequence) == self.frames_per_sample:
                    self.sequences.append(tuple(sequence))

        self.train = train
        
        if self.train:
            portion = 0.95
        else:
            portion = -0.05

        if portion < 0:
            self.sequences = self.sequences[int(len(self.sequences)*(1+portion)):]
        else:
            self.sequences = self.sequences[:int(len(self.sequences)*portion)]
        print(f"Dataset length: {self.__len__()}")
        random.shuffle(self.sequences)


    def __getitem__(self, index):
        sequence = self.sequences[index]
        query = []
        prediction = []
        for i, file in enumerate(sequence):
            with Image.open(file) as img:
                img = img.convert("L")
                img = img.resize((self.img_resolution,self.img_resolution))
                img = T.ToTensor()(img)* 255
                if self.predictioner:
                    if i < self.frames_per_sample//2:
                        query.append(img)
                    else:
                        prediction.append(img)
                else:
                    query.append(img)
 
        query = torch.stack(query)
        if self.predictioner:
            prediction = torch.stack(prediction)
        
        #print(query.shape, "expected", (self.frames_per_sample, 1, self.img_resolution, self.img_resolution))
        return query, prediction
        #return rearrange(vid, 'c t h w -> t c h w'), index


    def __len__(self):
        return len(self.sequences)


def get_loaders(imgstr, resolution, timesteps, skip, batch_size=1, n_gpus=1, seed=42,  cond=False, use_train_set=False, data_location=None, predictioner=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    if data_location is None:
        raise ValueError("data_location must be specified")
    if imgstr == 'METEO':
        if cond:
            print("here condition is turned on")
            timesteps *= 2
        trainset = Meteoset(data_location, train=True, resolution=resolution, nframes=timesteps, cond=cond, predictioner=predictioner)
        print(len(trainset))
        testset = Meteoset(data_location, train=False, resolution=resolution, nframes=timesteps, cond=cond, predictioner=predictioner)
        print(len(testset))
    else:
        raise NotImplementedError()    



    trainset_sampler = InfiniteSampler(dataset=trainset, num_replicas=n_gpus, seed=seed)
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=8, prefetch_factor=2)
    
    testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus,  seed=seed)
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size // n_gpus, pin_memory=False, num_workers=8, prefetch_factor=2)

    return trainloader, trainloader, testloader 


