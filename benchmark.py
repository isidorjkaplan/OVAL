import numpy as np
import cv2
import os
import time

from model import Autoencoder, Encoder, Decoder
from torch.utils.tensorboard import SummaryWriter

import torchvision
import cv2
import torch
import numpy as np
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import time
import argparse
import itertools
import glob, os
from loaders import VideoDatasetLoader
from ast import literal_eval
import subprocess
import torch

class Nothing(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return self.decode(self.encode(x))
    
    def encoder(self, x, hidden=None):
        return x

    def decoder(self, x, hidden=None):
        return x

class stupidEncoder(nn.Module):
    def __init__(self, save_path = None):
        super().__init__()
        #all we have is a maxpool layer that can shrink the image 4x
        self.halfing_pool = nn.MaxPool2d(2,2)
        self.save_path = save_path
    
    def forward(self, x):
        return self.decode(self.encode(x))

    def encoder(self, x, hidden=None):
        x = self.halfing_pool(x) #1/4
        x = self.halfing_pool(x) #1/16
        return x
    
    def decoder(self, x, hidden=None):
        h = 4 * x.shape[-2]
        w = 4 * x.shape[-1]

        x.resize_(x.shape[0], x.shape[1], h, w)
        return x

class MostSignificantOnlyEncoder(nn.Module):
    def __init__(self, bits_to_cut = 6):
        super().__init__()
        self.bits_to_cut = bits_to_cut

    def forward(self, x):
        return self.decode(self.encode(x))

    def encoder(self, x, hidden=None):

        #convert x to numpy array
        x = (x.numpy()*255).astype(int)

        #0 out specified bits
        x = x & ~((2**self.bits_to_cut)-1)

        #back to tensor
        x = torch.Tensor(x)
        return x #return
    
    def decoder(self, x, hidden=None):
        return torch.Tensor(x.numpy().astype(float)/255)

class ResizingEncoder(nn.Module):
    def __init__(self, factor=20):
        super().__init__()
        self.factor = factor
    
    def forward(self, x):
        return self.decode(self.encode(x))

    def encoder(self, x, hidden=None):
        #convert to numpy to do resize on cv2
        x = x.numpy()

        #get required h and w and allocate empty array
        h = int((1.0/self.factor) * x.shape[-2])
        w = int((1.0/self.factor) * x.shape[-1])
        newx = np.empty((x.shape[0], x.shape[1], w, h))

        #fill array
        outputsize = (h, w)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                newx[i, j] = cv2.resize(x[i, j], outputsize)

        #turn newx into tensor and return
        newx = torch.Tensor(newx)
        return newx
    
    def decoder(self, x, hidden=None):
        #convert to numpy to do resize on cv2
        x = x.numpy()

        #get required h and w and allocate empty array
        h = self.factor * x.shape[-2]
        w = self.factor * x.shape[-1]
        newx = np.empty((x.shape[0], x.shape[1], w, h))

        #fill array
        outputsize = (h, w)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                newx[i, j] = cv2.resize(x[i, j], outputsize)

        #turn newx into tensor and return
        newx = torch.Tensor(newx)
        return newx

#This function will parse terminal inputs from the user and then perform offline training
def main_benchmark():
    #cd "C:/Users/isido/OneDrive/Files/School/Year 3/Winter 2022/APS360/OVAL"
    #python3 offline.py --cuda --stop=160 --save_model=data/models/offline.pt --max_frames=100
    parser = argparse.ArgumentParser(description='Arguments for Online Training')
    parser.add_argument('--video_folder', default="data/videos", help='Path to a folder of videos to train on')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for the model')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--batch_size', type=int, default=30, help='Number of frames per training batch')
    parser.add_argument('--loss', default='bce', help='Loss function:  {mae, mse, bce} ')
    parser.add_argument("--load_model", default=None, help="File for the model to load")
    parser.add_argument("--max_frames", default=None, type=int, help="If specified, it will clip all videos to this many frames")
    parser.add_argument("--img_size", default="(480,360)", help="The dimensions for the image. Will be resized to this.")
    parser.add_argument("--color_space", default="bgr", help="the color space to use during training.")
    parser.add_argument("--benchmark", default=None, help="Choose which benchmark to use. {nothing, resize, cutbits}")
    parser.add_argument('--enc_bytes', type=int, default=16, help="Number of bytes per encoded element. {16, 32, 64}")
    args = parser.parse_args()

    video_size = literal_eval(args.img_size)

    #Select the device
    assert not args.cuda or torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    enc_bytes = {16:torch.float16, 32:torch.float32, 64:torch.float64}[args.enc_bytes]
    type_sizes = {torch.float16:2, torch.float32:4, torch.float64:8}

    #Select the loss function
    loss_fn = {'mse':F.mse_loss, 'mae':F.l1_loss, 'bce':nn.BCELoss()}[args.loss]

    assert (args.benchmark is None)^(args.load_model is None)
    #Load the model

    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model = Autoencoder(video_size)
        model.load_state_dict(torch.load(args.load_model))
    else:
        model = {"nothing":Nothing(), "resize":ResizingEncoder(), "cutbits":MostSignificantOnlyEncoder()}[args.benchmark]

    if args.cuda:
        print("Sending model to CUDA")
        model.to(device)

    #Create summary writer and print bookkeeping items
    start_time = time.time()
    writer = SummaryWriter("runs/test/%d" % start_time)
    writer.add_text("Model", str(model).replace("\n", "  \n"))
    arg_str = ""
    for key in vars(args):
        arg_str = "%s**%s**: %s  \n" % (arg_str, key, str(vars(args)[key])) 
    writer.add_text("Args", arg_str)
    writer.add_text("Git", subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip())

    epoch_loss = []
    test_loader = VideoDatasetLoader(os.path.join(args.video_folder, "test"), args.batch_size, max_frames=args.max_frames, video_size=video_size, color_space=args.color_space)
    test_loader.reset()
    iter_num = 0
    hidden_states = [[None,None] for video in test_loader.video_loaders]
    for data in test_loader:
        if data is None:
            break
        video_num, frames = data

        if args.load_model is not None:
            #Run the actual model
            frames = frames.to(device)
            enc_frames, hidden_states[video_num][0] = model.encoder(frames, hidden_states[video_num][0])
            prev_type = enc_frames.dtype
            enc_frames=enc_frames.type(enc_bytes)
            frames_out, hidden_states[video_num][1] = model.decoder(enc_frames.type(prev_type), hidden_states[video_num][1])
        else:
            enc_frames = model.encoder(frames)
            frames_out = model.decoder(enc_frames)

        uncomp_size = frames.numel() #Each is 1 byte in RGB normally
        comp_size = enc_frames.numel()*type_sizes[enc_frames.dtype]
        writer.add_scalar("Iter/train_comp_factor", uncomp_size/comp_size, iter_num)
        if (iter_num == 0):
            print("Compression Factor: %g" % (uncomp_size/comp_size))

        frames = frames[:,:,:frames_out.shape[2], :frames_out.shape[3]]
        frames_out = frames_out[:,:,:frames.shape[2],:frames.shape[3]]
        
        loss_v = loss_fn(frames_out, frames)
        epoch_loss.append(loss_v.item())
        writer.add_scalar("Iter/test_loss", loss_v.item(), iter_num)
        print("%d: Video=%d, Frames=%d/%d=%d, loss_test=%g" % (iter_num, video_num, test_loader.num_frames_read, test_loader.total_num_frames, 100*test_loader.num_frames_read/test_loader.total_num_frames, loss_v.item()))
            
        iter_num += 1
    
    loss = np.mean(epoch_loss)
    print("Total Test Loss: %g", loss)
    writer.add_text("Test Loss", str(loss))

    pass


if __name__ == "__main__":
    main_benchmark()
