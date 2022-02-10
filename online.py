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
from loaders import VideoSimulator, CameraVideoSimulator
from ast import literal_eval

from torch.utils.tensorboard import SummaryWriter
from model import Autoencoder, Encoder, Decoder

def linear_reward_func(enc_size, loss):
    return None #Do not use this yet

def print_thread(args, data_q, model):
    board = "runs/online/%d" % time.time()
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
    #cd "C:/Users/isido/OneDrive/Files/School/Year 3/Winter 2022/APS360/OVAL"
    #python3 online.py --video=data/videos/train/lwt_short.mp4 --stop=60 --cuda --load_model=data/models/offline.pt --out=data/videos/out/online.mp4
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
    parser.add_argument("--img_size", default="(480,360)", help="The dimensions for the image. Will be resized to this.")

    args = parser.parse_args()

    video_size = literal_eval(args.img_size)

    assert args.enc_bytes in [16, 32, 64]

    data_q = Queue()
    model = Autoencoder(video_size, save_path=args.save_model)
    if args.load_model is not None:
        print("Loading model: %s" % args.load_model)
        model.load_state_dict(torch.load(args.load_model))

        
    p = Process(target=print_thread, args=(vars(args), data_q,model,))
    p.start()
    # Download the sample video
    
    #shutil.rmtree(board)
    loss_fn = {'mse':F.mse_loss, 'mae':F.l1_loss}[args.loss]
    device = 'cuda' if args.cuda else 'cpu'
    enc_bytes = {16:torch.float16, 32:torch.float32, 64:torch.float64}[args.enc_bytes]
    sender = arch.Sender(model, linear_reward_func, data_q, enc_bytes=enc_bytes, loss_fn=loss_fn, lr=args.lr, max_buffer_size=args.buffer_size,update_threshold=args.update_err, live_device=device, train_device=device)

    if args.video is not None:
        video_sim = VideoSimulator(args.video, repeat=args.repeat_video, rate=args.fps, video_size=video_size)#, size=(340, 256))
    else:
        video_sim = CameraVideoSimulator(rate=args.fps, video_size=video_size)
    local_sim = sim.SingleSenderSimulator(sender, data_q)
    local_sim.start(video_sim, args.stop, args.out, loss_fn)
    p.kill()
    p.join()

    pass


if __name__ == "__main__":
    main_online()