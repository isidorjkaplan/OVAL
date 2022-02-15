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
class Autoencoder(nn.Module):
    def __init__(self, save_path=None):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_path = save_path

    def clone(self):
        ae = Autoencoder()
        ae.encoder.load_state_dict(self.encoder.state_dict())
        ae.decoder.load_state_dict(self.decoder.state_dict())
        return ae

    def save_model(self):
        if self.save_path is not None:
            torch.save(self.state_dict(), self.save_path)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to(self, device):
        self.encoder.to(device)
        self.decoder.to(device)

    #TODO load_state_dict
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 5,kernel_size=2,stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 8,kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8,6,kernel_size=8,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6,5,kernel_size=2,padding=1)
        )

        pass

    def forward(self, x):
        return self.conv_net(x)
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net_t = nn.Sequential(
            nn.ConvTranspose2d(5,6,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6,8,kernel_size=8,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 5,kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5,3,kernel_size=2,stride=1)
        )
        pass

    def forward(self, x):
        x = self.conv_net_t(x)
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
    parser.add_argument('--update_err', type=float, default=None, help='The ratio of losses error that causes a new model to be broadcast (0-1). Leave empty to send update every itteration (faster training as well since no local eval)')
    parser.add_argument('--stop', type=float, default=None, help='Time after which we stop video')
    parser.add_argument('--repeat_video', action="store_true", default=False, help='Repeat when the video runs out')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--enc_bytes', type=int, default=16, help="Number of bytes per encoded element. {16, 32, 64}")
    parser.add_argument('--buffer_size', type=int, default=20, help='The target buffer size in frames')
    parser.add_argument('--loss', default='mse', help='Loss function:  {mae, mse} ')
    parser.add_argument('--out', type=str, default=None, help='The path to save the decoded video for inspection')
    parser.add_argument("--load_model", default=None, help="File for the model to load")
    parser.add_argument("--save_model", default=None, help="File to save the model")
    parser.add_argument("--live_video", action="store_true", default=False, help="Turns on the real/decoded live video feed")
    parser.add_argument("--batch_size", type=int, default=5, help="Sets the batch size to be used in sending/receiving")
    parser.add_argument("--downsample", type=int, default=10000, help="The buffer size of the receive queue, after which we downsample")


    args = parser.parse_args()

    assert args.enc_bytes in [16, 32, 64]

    data_q = Queue()
    model = Autoencoder(save_path = args.save_model)
    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))

        
    p = Process(target=print_thread, args=(vars(args), data_q,model,))
    p.start()
    # Download the sample video
    
    #shutil.rmtree(board)
    loss_fn = {'mse':F.mse_loss, 'mae':F.l1_loss}[args.loss]
    device = 'cuda' if args.cuda else 'cpu'
    enc_bytes = {16:torch.float16, 32:torch.float32, 64:torch.float64}[args.enc_bytes];
    sender = arch.Sender(model, linear_reward_func, data_q, enc_bytes=enc_bytes, loss_fn=loss_fn, lr=args.lr, max_buffer_size=args.buffer_size,update_threshold=args.update_err, live_device=device, train_device=device)

    frameWidth, frameHeight = None, None
    if args.video is not None:
        video_sim = sim.VideoSimulator(args.video, repeat=args.repeat_video, rate=args.fps)#, size=(340, 256))
    else:
        video_sim = sim.CameraVideoSimulator(rate=args.fps)

    local_sim = sim.SingleSenderSimulator(sender, data_q, live_video=args.live_video)
    local_sim.start(video_sim, args.stop, args.out, args.fps, 5, args.downsample, loss_fn)
    p.kill()
    p.join()

    pass


if __name__ == "__main__":
    main_online()