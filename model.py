from grpc import xds_channel_credentials
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from collections import namedtuple
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from convLSTM import ConvLSTM

ConvSettings = namedtuple("ConvSettings", "out_channels stride kern")

#TEMPORRY, WILL REPLACE WITH RYANS AUTOENCODER MODEL LATER
class Autoencoder(nn.Module):
    def __init__(self, image_dim, n_channels=3, save_path=None, conv_settings=[ConvSettings(5,2,2), ConvSettings(2,2,2)], linear_layers=None):
        super().__init__()
        self.encoder = Encoder(image_dim, n_channels, conv_settings, linear_layers)
        self.decoder = Decoder(image_dim, n_channels, conv_settings, linear_layers, self.encoder.flatten_size, self.encoder.conv_out_shape)
        self.save_path = save_path

        print(self)

    def clone(self):
        return copy.deepcopy(self)

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
    def __init__(self, image_dim, n_channels, conv_settings, linear_layers):
        super().__init__()
        self.conv_net = [nn.Conv2d(n_channels, conv_settings[0].out_channels, kernel_size=conv_settings[0].kern, stride=conv_settings[0].stride)]
        [self.conv_net.extend([nn.ReLU(inplace=True), nn.Conv2d(conv_settings[i-1].out_channels, conv_settings[i].out_channels, kernel_size=conv_settings[i].kern, stride=conv_settings[i].stride)]) for i in range(1, len(conv_settings))]

        self.conv_net = nn.Sequential(*self.conv_net)
        self.image_dim = image_dim

        self.conv_out_shape = self.conv_net(torch.zeros(1, n_channels, image_dim[0], image_dim[1]))
        self.flatten_size = self.conv_out_shape.numel()
        self.conv_out_shape = self.conv_out_shape.shape[1:]
        self.has_linear = linear_layers is not None
        print(self.flatten_size)
        # self.rnn = nn.LSTM(self.flatten_size, self.flatten_size, 1, batch_first=True)
        self.convLSTM = ConvLSTM(conv_settings[-1].out_channels, 2, 3, padding=0, activation='relu', frame_size=image_dim)

        if self.has_linear:
            layers = [nn.Linear(self.flatten_size, linear_layers[0])]
            [layers.extend([nn.ReLU(inplace=True), nn.Linear(linear_layers[i-1], linear_layers[i])]) for i in range(1, len(linear_layers))]

            self.linear_net = nn.Sequential(*layers)


    def forward(self, x, hidden=None):
        assert (x.shape[2], x.shape[3]) == self.image_dim
        x = self.conv_net(x)
        # print(x.shape)
        # x = x.view(x.shape[0], 1, self.flatten_size)
        # print(x.shape)
        if hidden is not None:
            x = self.convLSTM(x, hidden)
        else:
            x = self.convLSTM(x, 0)

        if self.has_linear:
            x = self.linear_net(x)
        print("out of convLSTM", x.shape)
        return x

class Decoder(nn.Module):
    def __init__(self, image_dim, n_channels, conv_settings, linear_layers, flatten_size, conv_shape):
        super().__init__()
        self.has_linear = linear_layers is not None
        if self.has_linear:
            layers = []
            [layers.extend([nn.Linear(linear_layers[i], linear_layers[i-1]), nn.ReLU(inplace=True)]) for i in range(len(linear_layers)-1, 0, -1)]
            layers.extend([nn.Linear(linear_layers[0], flatten_size)])
            self.linear_net = nn.Sequential(*layers)
        print("flatten size: ",flatten_size)
        # self.rnn = nn.LSTM(flatten_size, flatten_size, 1, batch_first=True)
        self.convLSTM = ConvLSTM(conv_settings[-1].out_channels, 2, 3, padding=0, activation='relu', frame_size=image_dim)
        self.flatten_size = flatten_size
        self.conv_net = []
        [self.conv_net.extend([nn.ConvTranspose2d(conv_settings[i].out_channels, conv_settings[i-1].out_channels, kernel_size=conv_settings[i].kern, stride=conv_settings[i].stride), nn.ReLU(inplace=True)]) for i in range(len(conv_settings)-1, 0, -1)]
        self.conv_net.extend([nn.ConvTranspose2d(conv_settings[0].out_channels, n_channels, kernel_size=conv_settings[0].kern, stride=conv_settings[0].stride)])
        self.conv_net = nn.Sequential(*self.conv_net)

        self.image_dim = image_dim
        self.conv_shape = conv_shape



    def forward(self, x, hidden=None):
        if self.has_linear:
            print("in linear")
            x = self.linear_net(x)
            x = x.view(x.shape[0], 1, *self.conv_shape)
        print("decoder")
        # print(x.shape)

        if hidden is not None:
            x = self.convLSTM(x, hidden)
        else:
            x = self.convLSTM(x, 0)
        # print(x.shape)
        # x = x.view(x.shape[0], *self.conv_shape)
        # print(x.shape)
        x = self.conv_net(x)
        x = F.sigmoid(x)
        return x