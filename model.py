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
    def __init__(
        self, image_dim, num_enc_layers, n_channels=3, save_path=None,
        conv_settings=[ConvSettings(6, 1, 3), ConvSettings(8, 1, 3), ConvSettings(9, 2, 3), ConvSettings(10, 2, 4), ConvSettings(10, 2, 4),ConvSettings(10, 2, 4), ConvSettings(10, 2, 3)],
        enc_conv=[ConvSettings(4,2,3), ConvSettings(8,1,3), ConvSettings(10,2,3), ConvSettings(10,1,3), ConvSettings(10,2,3)],
        linear_layers=None
        ):
        super().__init__()
        self.encoder = Encoder(image_dim, n_channels, conv_settings, linear_layers, num_enc_layers)
        self.decoder = Decoder(image_dim, n_channels, conv_settings, linear_layers, self.encoder.flatten_size, self.encoder.conv_out_shape, num_enc_layers)
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
    def __init__(self, image_dim, n_channels, conv_settings, linear_layers, num_enc_layers, enc_conv):
        super().__init__()
        self.conv_net = [nn.Conv2d(n_channels, conv_settings[0].out_channels, kernel_size=conv_settings[0].kern, stride=conv_settings[0].stride)]
        [self.conv_net.extend([nn.ReLU(inplace=True), nn.Conv2d(conv_settings[i-1].out_channels, conv_settings[i].out_channels, kernel_size=conv_settings[i].kern, stride=conv_settings[i].stride)]) for i in range(1, len(conv_settings))]

        self.conv_net = nn.Sequential(*self.conv_net)
        self.image_dim = image_dim

        self.conv_out_shape = self.conv_net(torch.zeros(1, n_channels, image_dim[0], image_dim[1]))
        self.flatten_size = self.conv_out_shape.numel()
        self.conv_out_shape = self.conv_out_shape.shape[1:]
        self.has_linear = linear_layers is not None

        #def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        self.convLSTM = ConvLSTM(conv_settings[-1].out_channels, conv_settings[-1].out_channels, (3,3), num_layers=1)

        if self.has_linear:
            layers = [nn.Linear(self.flatten_size, linear_layers[0])]
            [layers.extend([nn.ReLU(inplace=True), nn.Linear(linear_layers[i-1], linear_layers[i])]) for i in range(1, len(linear_layers))]

            self.linear_net = nn.Sequential(*layers)
        self.num_enc_layers = num_enc_layers
        self.enc_conv = [nn.Sequential(nn.Conv2d(conv_settings[-1].out_channels,kernel_size=enc_conv[0].kern, stride=enc_conv[0].stride), nn.ReLU(inplace=True))]
        ##[self.enc_conv.extend([nn.ReLU(inplace=True), nn.Conv2d(enc_conv[i-1].out_channels, enc_conv[i].out_channels, kernel_size=enc_conv[i].kern, stride=enc_conv[i].stride)]) for i in range(1, len(enc_conv))]
        for i in range(1, len(enc_conv)):
            self.enc_conv.append(nn.Sequential(nn.Conv2d(enc_conv[i-1].out_channels, enc_conv[i].out_channels, kernel_size=enc_conv[i].kern, stride=enc_conv[i].stride), nn.ReLU(inplace=True)))

    def forward(self, x, hidden=None):
        assert (x.shape[2], x.shape[3]) == self.image_dim
        x = self.conv_net(x)

        if hidden is not None:
            hidden = [[x.detach() for x in y] for y in hidden]
        x, hidden = self.convLSTM(x.view(1, *x.shape), hidden)
        x = x[-1][0] #last conv layer, only one item in the batch (with a large sequence)
        #x = x[-1][:,0,:,:,:]


        if self.has_linear:
            x = self.linear_net(x.view(x.shape[0], self.flatten_size))
        for i in range(self.num_enc_layers):
            x = self.enc_conv[i](x)
        return x, hidden


class Decoder(nn.Module):
    def __init__(self, image_dim, n_channels, conv_settings, linear_layers, flatten_size, conv_shape, num_enc_layers):
        super().__init__()
        self.has_linear = linear_layers is not None
        if self.has_linear:
            layers = []
            [layers.extend([nn.Linear(linear_layers[i], linear_layers[i-1]), nn.ReLU(inplace=True)]) for i in range(len(linear_layers)-1, 0, -1)]
            layers.extend([nn.Linear(linear_layers[0], flatten_size)])
            self.linear_net = nn.Sequential(*layers)

        #self.convLSTM = ConvLSTM(conv_settings[-1].out_channels, conv_settings[-1].out_channels, 3, padding=0, activation='relu', frame_size=image_dim)
        self.convLSTM = ConvLSTM(conv_settings[-1].out_channels, conv_settings[-1].out_channels, (3,3), num_layers=1)
        self.flatten_size = flatten_size

        self.conv_net = []
        [self.conv_net.extend([nn.ConvTranspose2d(conv_settings[i].out_channels, conv_settings[i-1].out_channels, kernel_size=conv_settings[i].kern, stride=conv_settings[i].stride), nn.ReLU(inplace=True)]) for i in range(len(conv_settings)-1, 0, -1)]
        self.conv_net.extend([nn.ConvTranspose2d(conv_settings[0].out_channels, n_channels, kernel_size=conv_settings[0].kern, stride=conv_settings[0].stride)])
        self.conv_net = nn.Sequential(*self.conv_net)

        self.image_dim = image_dim
        self.conv_shape = conv_shape
        self.num_enc_layers = num_enc_layers

    def forward(self, x, hidden=None):
        if self.has_linear:
            x = self.linear_net(x)
            x = x.view(x.shape[0], 1, *self.conv_shape)

        if hidden is not None:
            hidden = [[x.detach() for x in y] for y in hidden]
        x, hidden = self.convLSTM(x.view(1, *x.shape), hidden)
        x = x[-1][0]
        #x = x[-1][:,0,:,:,:]

        x = self.conv_net(x)
        x = F.sigmoid(x)
        return x, hidden