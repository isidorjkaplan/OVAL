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
import time
import argparse

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
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)  
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(8, 3, 3, stride=2, padding=1)
        pass

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_conv4 = nn.ConvTranspose2d(3, 8, 2, stride=2, padding=1)#Using 3 to ensure larger then input
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1)
        pass

    def forward(self, x):
        x = F.relu(self.t_conv4(x))
        #x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv2(x))
        x = F.sigmoid(self.t_conv1(x))
        return x

def linear_reward_func(enc_size, loss):
    return None #Do not use this yet



def print_thread(args, data_q, model):
    board = "runs/%d" % time.time()
    writer = SummaryWriter(board)
    writer.add_text("Model/Encoder", str(model.encoder).replace("\n", "  \n"))
    writer.add_text("Model/Decoder", str(model.decoder).replace("\n", "  \n"))
    arg_str = ""
    for key in args:
        arg_str = "%s**%s**: %s  \n" % (arg_str, key, str(args[key])) 
    writer.add_text("Args", arg_str)
    
    while True:
        label, y, x = data_q.get()
        writer.add_scalar(label, y, x)
        

#This function will parse terminal inputs from the user and then perform online training
# It will load the model from the file specified by the user inputs
# It will setup the model as well as the simulator settings and tensorboard and all that
# It will then pass control to the simulator which will start all it's respective threads and begin running
def main_online():
    
    parser = argparse.ArgumentParser(description='Arguments for Online Training')
    parser.add_argument('--video', type=None, help='The path to the video to load (from current directory). If this is empty then uses the video camera instead.')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate for the model')
    parser.add_argument('--fps', type=float, default=40, help='The FPS to target (may be slower)')
    parser.add_argument('--update_err', type=float, default=0.08, help='The error that causes a new model to be broadcast')
    parser.add_argument('--stop', type=float, default=None, help='Time after which we stop video')
    parser.add_argument('--repeat_video', action="store_true", default=False, help='Repeat when the video runs out')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--buffer_size', type=int, default=20, help='The target buffer size in frames')
    parser.add_argument('--out', type=str, default=None, help='The path to save the decoded video for inspection')

    args = parser.parse_args()


    data_q = Queue()
    model = Autoencoder()
    p = Process(target=print_thread, args=(vars(args), data_q,model,))
    p.start()
    # Download the sample video
    
    #shutil.rmtree(board)
    device = 'cuda' if args.cuda else 'cpu'
    sender = arch.Sender(model, linear_reward_func, data_q, lr=args.lr, max_buffer_size=args.buffer_size,update_threshold=args.update_err, live_device=device, train_device=device)

    if args.video is not None:
        video_sim = sim.VideoSimulator(args.video, repeat=args.repeat_video, rate=args.fps)#, size=(340, 256))
    else:
        video_sim = sim.CameraVideoSimulator(rate=args.fps)
    local_sim = sim.SingleSenderSimulator(sender, data_q)
    local_sim.start(video_sim, args.stop, args.out)
    p.kill()
    p.join()

    pass


if __name__ == "__main__":
    main_online()