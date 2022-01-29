from collections import namedtuple

Autoencoder = namedtuple("Autoencoder", "encoder decoder")

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, features):
        pass

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, features):
        pass