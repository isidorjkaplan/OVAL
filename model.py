from collections import namedtuple
from tkinter import X
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from ConvLSTM import ConvLSTMCell

#Autoencoder = namedtuple("Autoencoder", "encoder decoder")

# Mode = ['Cutoff'] TODO, maybe also try dropout version
# Enc_Sizes = Array of possible encoding size choices
class Autoencoder(nn.Module):
    def __init__(self, num_enc_layers):
        super().__init__()
        self.encoder = Encoder(num_enc_layers)
        self.decoder = Decoder(num_enc_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class Encoder(nn.Module):
    def __init__(self, num_enc_layers:int, num_frames=2):
        super().__init__()
        self.num_enc_layers = num_enc_layers
        # feature extraction taken from first few layers of VGG 16
        self.conv_preproc = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,8,kernel_size=(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,padding=0,dilation=1,ceil_mode=False),
            nn.Conv2d(8,3,kernel_size=(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,padding=0,dilation=1, ceil_mode=False)
        )

        # ConvLSTM
        # self.encoder_1_convlstm = ConvLSTMCell(input_dim=3,
        #                                        hidden_dim=num_frames,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        # self.encoder_2_convlstm = ConvLSTMCell(input_dim=num_frames,
        #                                        hidden_dim=num_frames,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        # DownSampling
        self.downsampling = [
            nn.Sequential(nn.Conv2d(3,64,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(64,32,kernel_size=(3,3), padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(32,16,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(16,8,kernel_size=(3,3), padding=1), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(8,3,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True))
        ]

    #Takes in features as well as the encoding size to use this time
    # INPUTS:
    #    x = Input Image
    #    enc_size = Current encoding size to use
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    encoded image: The actual encoded image to the proper size
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, hidden=None, future_seq=0):
        """
        input: Tensor of shape (batch, time, channel, height, width)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        # b, seq_len, _, h, w = x.size()

        ## begin feature extraction
        x = self.conv_preproc(x)
        # b, seq_len, _, h, w = x.size()
        # ## conv lstm
        # h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, 0, :, :],
        #                                     cur_state=[h_t, c_t])  # we could concat to provide skip conn here
        # h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
        #                                         cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
        # x = h_t2

        # begin downsampling with variable encoding
        for i in range(self.num_enc_layers):
            x = self.downsampling[i](x)
        return x

class Decoder(nn.Module):
    def __init__(self, num_enc_layers:int, num_frames=2):
        super().__init__()
        self.num_enc_layers = num_enc_layers
        self.upsample = [
            nn.Sequential(nn.ConvTranspose2d(3,8,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(8,16,kernel_size=(3,3), padding=1), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(16,32,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(32,64,kernel_size=(3,3), padding=1), nn.ReLU(True)),
            nn.Sequential(nn.ConvTranspose2d(64,3,kernel_size=(3,3),stride=2, padding=1), nn.ReLU(True))
        ]
        # self.decoder_1_convlstm = ConvLSTMCell(input_dim=3,  # nf + 1
        #                                        hidden_dim=num_frames,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        # self.decoder_2_convlstm = ConvLSTMCell(input_dim=num_frames,
        #                                        hidden_dim=num_frames,
        #                                        kernel_size=(3, 3),
        #                                        bias=True)

        self.conv_postproc = nn.Sequential(
            nn.ConvTranspose2d(3,16,kernel_size=(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16,8,kernel_size=(3,3),padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8,3,kernel_size=(3,3),padding=1),
        )



     #Takes in encoded image as well as context / history
    # INPUTS:
    #    x = Encoded input image
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    decoded image: The actual decoded image, reconstructed
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, future_seq=0, hidden=None):
        for i in range(len(self.upsample)-self.num_enc_layers, len(self.upsample)):
            x = self.upsample[i](x)

        # b, seq_len, _, h, w = x.size()
        # h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=x,
        #                                         cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
        # h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
        #                                         cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
        # x = h_t4
        # outputs += [h_t4]  # predictions

        # outputs = torch.stack(outputs, 1)
        # outputs = outputs.permute(0, 2, 1, 3, 4)
        x = self.conv_postproc(x)
        # outputs = torch.nn.Sigmoid()(outputs)

        return x