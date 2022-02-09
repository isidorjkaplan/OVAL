
#TEMPORRY, WILL REPLACE WITH RYANS AUTOENCODER MODEL LATER
class Autoencoder(nn.Module):
    def __init__(self, save_path=None):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_path = save_path

    def clone(self):
        ae = Autoencoder()
        ae.encoder.load_state_dict(self.encoder.state_dict())
        ae.decoder.load_state_dict(self.decoder.state_dict())
        return ae

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
    def __init__(self):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 5,kernel_size=2,stride=2), 
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 8,kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8,6,kernel_size=8,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(6,5,kernel_size=2,padding=1)
        )

        pass

    def forward(self, x):
        return self.conv_net(x)
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net_t = nn.Sequential(
            nn.ConvTranspose2d(5,6,kernel_size=2,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(6,8,kernel_size=8,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 5,kernel_size=4, stride=2), 
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(5,3,kernel_size=2,stride=1)
        )
        pass

    def forward(self, x):
        x = self.conv_net_t(x)
        return x