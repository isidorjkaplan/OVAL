from collections import namedtuple
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
class Autoencoder():
    def __init__(self, num_enc_layers):
        self.encoder = Encoder("default", num_enc_layers)
        self.decoder = Decoder("default", num_enc_layers)


class Encoder(nn.Module):
    def __init__(self, mode:str, num_enc_layers:int, num_frames:int, input_dim):
        super().__init__()
        self.encoder_1_convlstm = ConvLSTMCell(input_dim=input_dim,
                                               hidden_dim=num_frames,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=num_frames,
                                               hidden_dim=num_frames,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.layers = [self.encoder_1_convlstm, self.encoder_1_convlstm]
    #Takes in features as well as the encoding size to use this time
    # INPUTS:
    #    x = Input Image
    #    enc_size = Current encoding size to use
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    encoded image: The actual encoded image to the proper size
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, enc_level:int, hidden=None, future_seq=0):
        """
        input: Tensor of shape (batch, time, channel, height, width)        #   batch, time, channel, height, width
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        assert enc_level < self.num_enc_layers

        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2
        return encoder_vector

class Decoder(nn.Module):
    def __init__(self, mode:str, num_enc_layers:int, num_frames):
        super().__init__()
        self.decoder_1_convlstm = ConvLSTMCell(input_dim=num_frames,  # nf + 1
                                               hidden_dim=num_frames,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=num_frames,
                                               hidden_dim=num_frames,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_CNN = nn.Conv3d(in_channels=num_frames,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.layers = [self.decoder_1_convlstm, self.decoder_2_convlstm, self.decoder_CNN]

     #Takes in encoded image as well as context / history
    # INPUTS:
    #    x = Encoded input image
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    decoded image: The actual decoded image, reconstructed
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, future_step, hidden=None):
        outputs = list()
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs