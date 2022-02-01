from collections import namedtuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

#Autoencoder = namedtuple("Autoencoder", "encoder decoder")

# Mode = ['Cutoff'] TODO, maybe also try dropout version
# Enc_Sizes = Array of possible encoding size choices
class Autoencoder():
    def __init__(self, num_enc_layers):
        self.encoder = Encoder(mode, num_enc_layers)
        self.decoder = Decoder(mode, num_enc_layers)


class Encoder(nn.Module):
    def __init__(self, mode:str, num_enc_layers:int, initial_dim):
        super().__init__()
        #Define convolutional layers for preprocessing
        self.preproc_conv_net = []  ## TODO
        ## TODO look into convutional LSTM, variable encoding
        #Define an RNN to hold the memory state, this is based on the convolutional features
        self.rnn = nn.Sequential(
            nn.LSTM(750, 1000, num_layers=3),
            nn.Linear(1000, 256),
            nn.Linear(256,128)
        ) # TODO

        #Define the autoencoder layers that reduce size down to enc_sizes[0]
        # This is some more potentially convolutional, not sure, preprocessing
        self.num_enc_layers = num_enc_layers #Save for use in forward
        #TODO Make sure to use CONVOLUTIONAL encoding layers, this will require some clever work

        encoded_space_dim = 5   ## change to actual dimension

        self.layers = nn.Sequential(
            nn.Conv2d(initial_dim[0],initial_dim[1],initial_dim[2],stride=2,padding=1),     ## dummy dimensions, fill in later
            nn.ReLU(True),
            nn.Conv2d(initial_dim[1],16,3, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(3*3*32,128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        ) #Todo, layers, define according to some formula

        pass

    #Takes in features as well as the encoding size to use this time
    # INPUTS:
    #    x = Input Image
    #    enc_size = Current encoding size to use
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    encoded image: The actual encoded image to the proper size
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, enc_level:int, hidden=None):
        assert enc_level < self.num_enc_layers
        #Extract convolutional features
        x = self.preproc_conv_net(x)
        #Perform RNN on feature extraction, adds some context
        x, hidden = self.rnn(x, hidden)
        #Peform autoencoder downscaling to encoded version

        # for i in range(0, enc_level): #Only run the first enc_level encoding layers
        #     x = F.relu(self.layers[i](x))
        for i in range(enc_level):
            x = self.layers[i](x)
        return x, hidden

class Decoder(nn.Module):
    def __init__(self, mode:str, num_enc_layers:int):
        super().__init__()
        #Define deconvolutional layers for reconstruction
        self.conv_net = None # TODO
        ## TODO pass encoding layers, variable encoding, convolutional LSTM
        #Define an RNN to hold the memory state, this is based on the convolutional features
        self.rnn = nn.Sequential(
            nn.LSTM(750, 1000, num_layers=3),
            nn.Linear(1000, 256),
            nn.Linear(256,128)
        ) # TODO

        #Define the autoencoder layers that increase size to enc_sizes[-1]
        # This is some more potentially convolutional, not sure, preprocessing
        initial_dim = [1,16,3]
        self.num_enc_layers = num_enc_layers
        self.layers = nn.Sequential(
            nn.Linear(num_enc_layers, 128),
            nn.ReLU(True),
            nn.Linear(128, 3*3*32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflatten=(32,3,3)),
            nn.ConvTranspose2d(32,16,3,stride=2,output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,3,stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(initial_dim[1],initial_dim[0],initial_dim[2],stride=2,padding=1,output_padding=1)
        ) #Todo, layers according to some formula

        pass

     #Takes in encoded image as well as context / history
    # INPUTS:
    #    x = Encoded input image
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    decoded image: The actual decoded image, reconstructed
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, hidden=None):
        #Extract from x which encoding layer was the last used, we then use that to reverse it here
        enc_level = None #TODO, extract from "x" to figure out its size
        #Perform upscaling from whatever initial size was,
        #we don't use the first enc_level layers since that is where we terminated encoding
        # for i, size in range(enc_level, self.num_enc_layers):
        #     x = F.relu(self.layers[i](x))
        for i,_ in range(enc_level, self.num_enc_layers):
            x = self.layers[i](x)
        #It is now been upscaled to appropriate size, call RNN to add context
        x, hidden = self.rnn(x, hidden)
        #Perform final convolutional inverse to get original image
        x = self.conv_net(x)
        #Return
        return x, hidden

        pass