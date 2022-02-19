
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
from loaders import VideoLoader
from skvideo.io import FFmpegWriter
from ast import literal_eval



def main_offline():
    #python3 convert_video.py data/models/offline.pt data/videos/other/lwt_short.mp4 data/videos/out/lwt_short.mp4 --cuda --max_frames=200
    parser = argparse.ArgumentParser(description='Encode and Decode a video using a trained model')
    parser.add_argument("load_model", help="File for the model to load")
    parser.add_argument('video', help='Path to an mp4 to convert')
    parser.add_argument('output', help="File to save the resulting converted video")
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--max_frames', type=int, default=None, help="Stop after fixed number of frames if set")
    parser.add_argument('--batch_size', type=int, default=30, help='Number of frames to pass through network at a time')
    parser.add_argument("--img_size", default="(480,360)", help="The dimensions for the image. Will be resized to this. Pass '(x,y)' ")
    args = parser.parse_args()

    #Select the device
    assert not args.cuda or torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    video_size = literal_eval(args.img_size)

    #Load the model
    model = Autoencoder(video_size)
    print("Loading model: %s" % args.load_model)
    model.load_state_dict(torch.load(args.load_model))
    if args.cuda:
        print("Sending model to CUDA")
        model.to(device)

    writer = FFmpegWriter(args.output)

    loader = VideoLoader(args.video, args.batch_size, max_frames=args.max_frames, video_size=video_size)

    for data in loader:
        if data is None:
            break
        frames, done = data

        print("Perctenage Complete: %d/%d=%d" % (loader.num_frames_read, loader.frameCount, 100*loader.num_frames_read/loader.frameCount))
        
        frames = frames.to(device)
        dec_frame = model(frames).detach().cpu()

        dec_np_frame = dec_frame.permute(0, 2, 3, 1).numpy()
        dec_np_frame = np.uint8(255*dec_np_frame)

        for i in range(len(dec_np_frame)):
            dec_np_frame[i] = cv2.cvtColor(dec_np_frame[i], cv2.COLOR_BGR2HSV)

        writer.writeFrame(dec_np_frame)

    writer.close()

    pass


if __name__ == "__main__":
    main_offline()