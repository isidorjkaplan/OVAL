
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

class VideoLoader():
    #Opens the file and initilizes the video
    def __init__(self, vid_id, filepath, batch_size, video_size=None):
        self.cap = cv2.VideoCapture(filepath)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.batch_size = batch_size
        self.video_size = video_size
        self.vid_id = vid_id
        
        #print("Loading Video=%s with frames=%d and size=(%d,%d)" % (filepath, self.frameCount, self.frameWidth, self.frameHeight))


    def __iter__(self):
        return self

    def __next__(self):
        if self.num_frames_read == self.frameCount:
            return None

        buf = np.empty((self.frameCount, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < self.batch_size and self.num_frames_read < self.frameCount  and ret):
            ret, buf[fc] = self.cap.read()
            if self.video_size is not None: #Optionally resize to specific size
                buf[fc] = cv2.resize(buf[fc], self.video_size, interpolation = cv2.INTER_LINEAR)
            fc += 1
            self.num_frames_read += 1

        if self.num_frames_read == self.frameCount:
            self.cap.release()

        return self.buf, self.num_frames_read == self.frameCount

    def __del__(self):
        pass


class VideoDatasetLoader():
    #Opens the file and initilizes the video
    def __init__(self, dir, batch_size, video_size=None):
        self.last_video = 0
        self.video_loaders = None

    def __iter__(self):
        return self

    def reset():
        print("Reset video loader dataset")
        if self.video_loaders is not None:
            del self.video_loaders
        self.video_loaders = [VideoLoader(vid_id, filepath)  for vid_id,filepath in enumerate(glob.glob("%s/*.mp4" % dir))]

    def __next__(self):
        if len(self.video_loaders) == 0:
            return None #No videos left for this epoch, must reset

        loader_num = self.last_video % len(self.video_loaders)
        video_loader = self.video_loaders[loader_num]
        vid_id = video_loader.vid_id
        frames, done = next(loader)
        if done:
            del self.video_loaders[loader_num]
        
        return vid_id, frames


#This function will parse terminal inputs from the user and then perform offline training
def main_offline():
    parser = argparse.ArgumentParser(description='Arguments for Online Training')
    parser.add_argument('--video_folder', type="data/videos", help='Path to a folder of videos to train on')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate for the model')
    parser.add_argument('--stop', type=float, default=None, help='Time after which we stop training, done or not')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs after which we stop training. ')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--batch_size', type=int, default=20, help='Number of frames per training batch')
    parser.add_argument('--loss', default='mse', help='Loss function:  {mae, mse} ')
    parser.add_argument("--load_model", default=None, help="File for the model to load")
    parser.add_argument("--save_model", default=None, help="File to save the model")
    args = parser.parse_args()

    #Select the device
    assert not args.cuda or torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    #Select the loss function
    loss_fn = {'mse':F.mse_loss, 'mae':F.l1_loss}[args.loss]

    #Load the model
    model = Autoencoder(save_path=args.save_model)
    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))
    if args.cuda:
        print("Sending model to CUDA")
        model.to(device)

    batch_size = args.batch_size
    epochs_iter = range(args.epochs) if args.epochs is not None else itertools.count(start=0)
    stop_time = (time.time() + args.stop) if args.stop is not None else None

    #Construct the optimizer
    optim = torch.optim.Adam(params, lr=args.lr)

    #Create summary writer and print bookkeeping items
    writer = SummaryWriter("runs/offline/%d" % time.time())
    writer.add_text("Model/Encoder", str(model.encoder).replace("\n", "  \n"))
    writer.add_text("Model/Decoder", str(model.decoder).replace("\n", "  \n"))
    arg_str = ""
    for key in args:
        arg_str = "%s**%s**: %s  \n" % (arg_str, key, str(args[key])) 
    writer.add_text("Args", arg_str)

    train_loader = VideoDatasetLoader(os.path.join(args.video_folder, "train"), args.batch_size)
    valid_loader = VideoDatasetLoader(os.path.join(args.video_folder, "valid"), args.batch_size)

    #Main training loop
    iter_num = 0
    for epoch in epochs_iter:
        train_loader.reset()
        valid_loader.reset()
        #Training loop
        epoch_loss = []
        for video_num, frames in train_loader:
            frames = frames.to(device)
            frames_out = model(frames)

            loss = loss_fn(frames, frames_out)

            loss.backward()
            optim.step()
            optim.zero_grad()

            #Bookkeeping items
            epoch_loss.append(loss.item())
            writer.add_scalar("Iter/train_loss", loss.item(), iter_num)
            iter_num+=1
        epoch_loss = np.mean(epoch_loss)

        #Validation Tracking
        valid_loss = []
        for video_nun, frames in valid_loader:
            frames = frames.to(device)
            frames_out = model(frames)
            loss = loss_fn(frames, frames_out)

            valid_loss.append(loss.item())
        valid_loss = np.mean(valid_loss)

        writer.add_scalar("Epochs/train_loss", epoch_loss, epoch)
        writer.add_scalar("Epochs/valid_loss", valid_loss, epoch)

    pass


if __name__ == "__main__":
    main_offline()