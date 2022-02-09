
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

#This function will parse terminal inputs from the user and then perform offline training
def main_offline():
    parser = argparse.ArgumentParser(description='Arguments for Online Training')
    parser.add_argument('--video_folder', type=None, help='Path to a folder of videos to train on')
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

    #Main training loop
    iter_num = 0
    for epoch in epochs_iter:
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