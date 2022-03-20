
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
import formater

import benchmark


def main_convert_video():
    #python3 convert_video.py data/models/offline.pt data/videos/other/lwt_short.mp4 data/videos/out/lwt_short.mp4 --cuda --max_frames=200
    parser = argparse.ArgumentParser(description='Encode and Decode a video using a trained model')
    parser.add_argument('video', help='Path to an mp4 to convert')
    parser.add_argument('output', help="File to save the resulting converted video")
    parser.add_argument("--load_model", help="File for the model to load")
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda')
    parser.add_argument('--max_frames', type=int, default=None, help="Stop after fixed number of frames if set")
    parser.add_argument('--batch_size', type=int, default=30, help='Number of frames to pass through network at a time')
    parser.add_argument("--img_size", default="(480,360)", help="The dimensions for the image. Will be resized to this. Pass '(x,y)' ")
    parser.add_argument("--color_space", default="bgr", help="color space used in training")
    parser.add_argument("--lstm", default=False, action='store_true', help="Add a conv LSTM layer. WARN: HUGE SLOWDOWN")
    parser.add_argument("--benchmark", default=None, help="Choose which benchmark to use. {nothing, resize, cutbits}")
    args = parser.parse_args()

    #Select the device
    assert not args.cuda or torch.cuda.is_available()
    device = 'cuda' if args.cuda else 'cpu'

    video_size = literal_eval(args.img_size)

    #Load the model
    assert (args.benchmark is None)^(args.load_model is None)
    #Load the model

    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model = Autoencoder(video_size, save_path=args.save_model)
        model.load_state_dict(torch.load(args.load_model))
    else:
        model = {"nothing":benchmark.Nothing(), "resize":benchmark.ResizingEncoder(), "cutbits":benchmark.MostSignificantOnlyEncoder()}[args.benchmark]

    if args.cuda:
        print("Sending model to CUDA")
        model.to(device)

    writer = FFmpegWriter(args.output)

    loader = VideoLoader(args.video, args.batch_size, max_frames=args.max_frames, video_size=video_size)

    hidden = [None, None]

    for data in loader:
        if data is None:
            break
        frames, done = data

        print("Perctenage Complete: %d/%d=%d" % (loader.num_frames_read, loader.frameCount, 100*loader.num_frames_read/loader.frameCount))
        
        frames = frames.to(device)
        if args.load_model is not None:
            enc_frames, hidden[0] = model.encoder(frames, hidden[0])
            dec_frame, hidden[1] = model.decoder(enc_frames, hidden[1])
        else:
            enc_frames = model.encoder(frames)
            dec_frame = model.decoder(enc_frames)

        dec_frame = dec_frame.detach().cpu()

        dec_np_frame = dec_frame.permute(0, 3, 2, 1).numpy()
        dec_np_frame = np.uint8(255*dec_np_frame)

        #for i in range(len(dec_np_frame)):
        #    dec_np_frame[i] = cv2.cvtColor(dec_np_frame[i], cv2.COLOR_BGR2HSV)

        formatter = formater.VideoFormatConverter(output_format=args.color_space)
        dec_np_frame = formatter.decode(dec_np_frame)

        for i in range(len(dec_np_frame)):
            dec_np_frame[i] = cv2.cvtColor(dec_np_frame[i], cv2.COLOR_BGR2RGB)

        writer.writeFrame(dec_np_frame)

    writer.close()

    pass


if __name__ == "__main__":
    main_convert_video()