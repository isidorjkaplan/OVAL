import torchvision
import cv2
import torch
import numpy as np

import simulator as sim
import architecture as arch
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing import Process, Value
from torch.multiprocessing import Queue
import shutil
import os

from torch.utils.tensorboard import SummaryWriter

#TEMPORRY, WILL REPLACE WITH RYANS AUTOENCODER MODEL LATER
class Autoencoder():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def clone(self):
        ae = Autoencoder()
        ae.encoder.load_state_dict(self.encoder.state_dict())
        ae.decoder.load_state_dict(self.decoder.state_dict())
        return ae

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    #TODO load_state_dict
    
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



def print_thread(board, data_q):
    writer = SummaryWriter(board)
    while True:
        label, y, x = data_q.get()
        writer.add_scalar(label, y, x)
        

#This function will parse terminal inputs from the user and then perform online training
# It will load the model from the file specified by the user inputs
# It will setup the model as well as the simulator settings and tensorboard and all that
# It will then pass control to the simulator which will start all it's respective threads and begin running
def main_online():
    # Download the sample video
    video_sim = sim.VideoSimulator('./data/lwt_short.mp4', repeat=False)#, size=(340, 256))
    #[frame for frame in video_sim]
    data_q = Queue()
    board = "runs/exp1"
    shutil.rmtree(board)
    p = Process(target=print_thread, args=("runs/exp1", data_q,))
    p.start()
    sender = arch.Sender(Autoencoder(), linear_reward_func, data_q)
    local_sim = sim.SingleSenderSimulator(sender, video_sim, data_q)
    local_sim.start()
    p.kill()
    p.join()

    pass


if __name__ == "__main__":
    main_online()