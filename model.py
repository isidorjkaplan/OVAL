import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

#TEMPORRY, WILL REPLACE WITH RYANS AUTOENCODER MODEL LATER
class Autoencoder(nn.Module):
    def __init__(self, image_dim, n_channels=3, save_path=None):
        super().__init__()
        self.encoder = Encoder(image_dim, n_channels)
        self.decoder = Decoder(image_dim, n_channels)
        self.save_path = save_path

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
    def __init__(self, image_dim, n_channels):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 5,kernel_size=2,stride=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 8,kernel_size=4, stride=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(8,6,kernel_size=8,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6,5,kernel_size=2,padding=1)
        )

        pass

    def forward(self, x):
        return self.conv_net(x)
        
class Decoder(nn.Module):
    def __init__(self, image_dim, n_channels):
        super().__init__()
        self.conv_net_t = nn.Sequential(
            nn.ConvTranspose2d(5,6,kernel_size=2,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6,8,kernel_size=8,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 5,kernel_size=4, stride=1), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5,3,kernel_size=2,stride=1)
        )
        pass

    def forward(self, x):
        x = self.conv_net_t(x)
        return x