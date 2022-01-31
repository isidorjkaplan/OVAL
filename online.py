import torchvision
import cv2
import torch
import numpy as np

import simulator as sim
import architecture as arch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

#TEMPORRY, WILL REPLACE WITH RYANS AUTOENCODER MODEL LATER
Autoencoder = namedtuple("Autoencoder", "encoder decoder")
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        pass

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)
        pass

    def forward(self, x):
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x

def linear_reward_func(enc_size, loss):
    return None #Do not use this yet

#This function will parse terminal inputs from the user and then perform online training
# It will load the model from the file specified by the user inputs
# It will setup the model as well as the simulator settings and tensorboard and all that
# It will then pass control to the simulator which will start all it's respective threads and begin running
def main_online():
    # Download the sample video
    video_sim = sim.VideoSimulator('./data/test.mp4')
    #[frame for frame in video_sim]
    board = "runs/exp1"
    sender = arch.Sender(Autoencoder(Encoder(), Decoder()), linear_reward_func, board)
    local_sim = sim.SingleSenderSimulator(sender, video_sim, board)
    local_sim.start()

    pass


if __name__ == "__main__":
    main_online()