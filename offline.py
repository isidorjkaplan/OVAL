#import libraries
import numpy as np
import pandas as pd
import cv2
import os
import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.nn import functional as f
from torch.utils.data import DataLoader
from multiprocessing import Process

#from torch.utils.start_tensorboard import run_tensorboard
from model import Autoencoder

#Simulates a file as being a live video stream returning rate frames per second
class videoLoader():
    #Opens the file and initilizes the video
    def __init__(self, filepath, rate=30, repeat = False, batch_size = 30):
        #getting video and then saving the details
        self.video = cv2.VideoCapture(filepath)
        rate = int(self.video.get(cv2.CAP_PROP_FPS))
        self.batch_size = batch_size
        self.frameCount = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Loading Video=%s with frames=%d and size=(%d,%d)" % (filepath, self.frameCount, self.frameWidth, self.frameHeight))
        
        #Parameters for frame reading
        self.num_frames_read = 0
        self.batches_read = 0
        self.last_frame_time = time.time()
        self.time_between_frames = 1.0/rate
        self.repeat = repeat

        buf = np.empty((batch_size, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        ret = True
        while (self.num_frames_read < (self.batches_read  + 1)* batch_size  and ret):
            ret, buf[self.num_frames_read] = self.video.read()
            self.num_frames_read += 1
        self.batches_read += 1
        #self.video.release()

        self.buffer = buf #save buffer
        #self.frames = torch.FloatTensor(buf)/255
        #self.frames = self.frames.permute(0, 3, 1, 2)#Make channel major

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
    
    def next_batch(self):
        #returns buffer and refills it to the next batch
        #Do all the reading and processing of the frame
        result = self.buffer
        buf = np.empty((self.batch_size, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        ret = True

        while (self.num_frames_read < (self.batches_read  + 1)* self.batch_size  and ret):
            ret, buf[(self.num_frames_read - (self.batches_read * self.batch_size))] = self.video.read()
            self.num_frames_read += 1
        self.batches_read += 1
        self.buffer = buf
        return result, ret

#Offline Training
# This function will train on large batches of data consisting of many videos
# The goal of offline training is to produce the most general possible model to start with
# Then each time a video call happens they retrain the offline model to overfit to their own scenario
# 
# INPUT: An autoencoder=(encoder/decoder), dataset=(train/validation) sets, and hyperparameters
#        Note for this training we train on a large volume of complete videos, this is the pre-training
# OUTPUT: Trained autoencoder, graph / history of training
#def train_offline(autoencoder, dataset, settings, board):

def train_offline(autoencoder, dataset, num_epochs):
    #Must make sure to evenly evaluate all autoencoder.enc_sizes to ensure it functions for all of them
    model = autoencoder
    for epoch in range(1, num_epochs + 1):
        losses = []
        end_video_not_reached = True
        while (end_video_not_reached):
            pass


#This function will parse terminal inputs from the user and then perform offline training
def main_offline():
    pass


if __name__ == "__main__":
    main_offline()

    loaded = videoLoader("../Movies/HuckleberryFinn.mp4")
    model = Autoencoder(num_enc_layers=5)
    videoNotEnd = True
    toyCounter = 0
    while (videoNotEnd):
        print(toyCounter)
        img, videoNotEnd = next(loaded)
        img = torch.from_numpy(img)
        img = torch.transpose(img, 1, 3)
        img = img.float()
        if (videoNotEnd):
            img_proc = model(img)
            print(img_proc.shape)
