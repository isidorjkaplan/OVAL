from collections import namedtuple

#Autoencoder = namedtuple("Autoencoder", "encoder decoder")

# Mode = ['Cutoff'] TODO, maybe also try dropout version
# Enc_Sizes = Array of possible encoding size choices
class Autoencoder():
    def __init__(self, enc_sizes=[8, 16, 32, 64]):
        self.encoder = Encoder(mode, enc_sizes)
        self.decoder = Decoder(mode, enc_sizes)
    

class Encoder(nn.Module):
    def __init__(self, mode, enc_sizes):
        super().__init__()
        #Define convolutional layers for preprocessing
        self.conv_net = None # TODO
        
        #Define an RNN to hold the memory state, this is based on the convolutional features
        self.rnn = None # TODO

        #Define the autoencoder layers that reduce size down to enc_sizes[0]
        # This is some more potentially convolutional, not sure, preprocessing
        self.enc_sizes = enc_sizes #Save for use in forward
        self.layers = [None] #Todo, layers should match enc_sizes
        
        pass
    
    #Takes in features as well as the encoding size to use this time
    # INPUTS: 
    #    x = Input Image
    #    enc_size = Current encoding size to use
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    encoded image: The actual encoded image to the proper size
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, enc_size, hidden=None):
        assert enc_size in self.enc_sizes #Must match a valid layer
        #Extract convolutional features
        x = self.conv_net(x)
        #Perform RNN on feature extraction, adds some context
        x, hidden = self.rnn(x, hidden)
        #Peform autoencoder downscaling to encoded version
        for i,size in self.enc_sizes:
            x = F.relu(self.layers[i](x))
            #Stop once we hit encoding size
            if size == enc_size:
                break
        return x, hidden

class Decoder(nn.Module):
    def __init__(self, mode, enc_sizes):
        super().__init__()
        #Define deconvolutional layers for reconstruction
        self.conv_net = None # TODO
        
        #Define an RNN to hold the memory state, this is based on the convolutional features
        self.rnn = None # TODO

        #Define the autoencoder layers that increase size to enc_sizes[-1]
        # This is some more potentially convolutional, not sure, preprocessing
        self.enc_sizes = enc_sizes #Save for use in forward
        self.layers = [None] #Todo, layers should match enc_sizes

        pass
    
     #Takes in encoded image as well as context / history
    # INPUTS: 
    #    x = Encoded input image
    #    hidden = Initial hidden state for LSTM. In training not used since we train on batch but used for evaluation
    # OUTPUTS:
    #    decoded image: The actual decoded image, reconstructed
    #    hidden:        Also returns the hidden state for future use if needed
    def forward(self, x, hidden=None):
        #Perform upscaling from whatever initial encoding size was
        enc_size = x.shape[1]
        assert enc_size in self.enc_sizes
        for i, size in self.enc_sizes:
            if size >= enc_size:
                x = F.relu(self.layers[i](x))
        #It is now been upscaled to appropriate size, call RNN to add context
        x, hidden = self.rnn(x, hidden)
        #Perform final convolutional inverse to get original image
        x = self.conv_net(x)
        #Return
        return x, hidden

        pass