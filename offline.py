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



#This function will parse terminal inputs from the user and then perform offline training
def main_offline():
    #cd "C:/Users/isido/OneDrive/Files/School/Year 3/Winter 2022/APS360/OVAL"
    #python3 offline.py --cuda --stop=160 --save_model=data/models/offline.pt --max_frames=100
    parser = argparse.ArgumentParser(description='Arguments for Online Training')
    parser.add_argument('--video_folder', default="data/videos", help='Path to a folder of videos to train on')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate for the model')
    parser.add_argument('--stop', type=float, default=None, help='Time after which we stop training, done or not')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs after which we stop training. ')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--batch_size', type=int, default=30, help='Number of frames per training batch')
    parser.add_argument('--loss', default='mse', help='Loss function:  {mae, mse} ')
    parser.add_argument("--load_model", default=None, help="File for the model to load")
    #parser.add_argument("--save_every", type=int, default=100, help="Save a copy of the model every N itterations")
    parser.add_argument("--save_model", default="data/models/offline.pt", help="File to save the model")
    parser.add_argument("--max_frames", default=None, type=int, help="If specified, it will clip all videos to this many frames")
    parser.add_argument("--img_size", default="(480,360)", help="The dimensions for the image. Will be resized to this.")
    args = parser.parse_args()

    video_size = literal_eval(args.img_size)

    #Select the device
    assert not args.cuda or torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    #Select the loss function
    loss_fn = {'mse':F.mse_loss, 'mae':F.l1_loss}[args.loss]

    #Load the model
    model = Autoencoder(video_size, save_path=args.save_model)
    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))
    if args.cuda:
        print("Sending model to CUDA")
        model.to(device)

    batch_size = args.batch_size
    epochs_iter = range(args.epochs) if args.epochs is not None else itertools.count(start=0)
    if args.stop is not None:
        stop_time = time.time() + args.stop

    #Construct the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    #Create summary writer and print bookkeeping items
    start_time = time.time()
    writer = SummaryWriter("runs/offline/%d" % start_time)
    writer.add_text("Model", str(model).replace("\n", "  \n"))
    arg_str = ""
    for key in vars(args):
        arg_str = "%s**%s**: %s  \n" % (arg_str, key, str(vars(args)[key])) 
    writer.add_text("Args", arg_str)

    train_loader = VideoDatasetLoader(os.path.join(args.video_folder, "train"), args.batch_size, max_frames=args.max_frames, video_size=video_size)
    valid_loader = VideoDatasetLoader(os.path.join(args.video_folder, "valid"), args.batch_size, max_frames=args.max_frames, video_size=video_size)

    #Main training loop
    best_val_loss = float('inf')
    iter_num = 0
    for epoch in epochs_iter:
        train_loader.reset()
        valid_loader.reset()
        #Training loop
        epoch_loss = []
        for data in train_loader:
            if data is None:
                break

            if args.stop is not None and time.time() > stop_time:
                print("Time ran out. Stopping.")
                return

            video_num, frames = data
            frames = frames.to(device)
            frames_out = model(frames)
            #Output does not exactly match size, truncate so that they are same size for loss. 
            frames = frames[:,:,:frames_out.shape[2], :frames_out.shape[3]]
            frames_out = frames_out[:,:,:frames.shape[2],:frames.shape[3]]
            #Run the loss function
            loss = loss_fn(frames, frames_out)

            loss.backward()
            optim.step()
            optim.zero_grad()

            #Bookkeeping items
            epoch_loss.append(loss.item())
            writer.add_scalar("Iter/train_loss", loss.item(), iter_num)

            #if iter_num % args.save_every == 0:
            #    model.save_model()
            print("%d: Frames=%d/%d=%d, loss_t=%g" % (iter_num, train_loader.num_frames_read, train_loader.total_num_frames, 100*train_loader.num_frames_read/train_loader.total_num_frames, loss.item()))
            iter_num+=1
        epoch_loss = np.mean(epoch_loss)

        #Validation Tracking
        valid_loss = []
        for data in valid_loader:
            if data is None:
                break
            video_num, frames = data
            
            frames = frames.to(device)
            frames_out = model(frames)

            frames = frames[:,:,:frames_out.shape[2], :frames_out.shape[3]]
            frames_out = frames_out[:,:,:frames.shape[2],:frames.shape[3]]
            
            loss = loss_fn(frames, frames_out)

            valid_loss.append(loss.item())
        valid_loss = np.mean(valid_loss)

        if valid_loss < best_val_loss:
            print("Saving model: loss_v=%g", valid_loss)
            best_val_loss = valid_loss
            model.save_model()


        writer.add_scalar("Epochs/train_loss", epoch_loss, epoch)
        writer.add_scalar("Epochs/valid_loss", valid_loss, epoch)
        print("Epoch %d: loss_t=%g, loss_v=%g" % (epoch, epoch_loss, valid_loss))

    pass


if __name__ == "__main__":
    main_offline()

    # loaded = videoLoader("../Movies/HuckleberryFinn.mp4")
    # for i in range(100):
    #     img = next(loaded)[5]
    #     cv2.imwrite(f"test/{i}.jpeg", img)