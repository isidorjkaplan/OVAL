import copy

#This class encompasses the sender. 
# DESC: Anything that is done at the sender is handled within this class
# INPUT: This class will take in frames, one at a time, representing the live video
#        Also takes in a pre-trained model from offline learning to be updated
# OUTPUT: This will output encoded frames, and periodically decoder models
class Sender():
    #Init function for Sender
    # INPUT: A pre-trained autoencoder pair from offline learning. Note, autoencoder = (encoder, decoder) pair
    def __init__(self, autoencoder):
        #Live model is used for actively encoding frames, and stores the last broadcast model
        self.live_model = copy.deepcopy(autoencoder)
        #Train model is the one that we are actively training. Periodicailly we set live_model = train_model with a broadcast
        self.train_model = autoencoder
        #The hidden state of our LSTM. Will carry over even as we switch active models. Used for the real-time frames
        self.hidden = None
        pass

    # PUBLIC METHODS
    
    #Run one iteration of training on its local buffer to train AE
    # Does not modify the hidden state of the LSTM used for live decoding
    # This function performs a step and does an update
    #
    # INPUT: Nothing. Trains on it's history of frames from evaluate calls. 
    # OUTPUT: Either returns "None" or a "Decoder" to be broadcast on the network. 
    def step(self):
        pass
    
    #Evaluate a single frame. 
    # This will use previous evaluate call hidden state for RNN
    # Each sequence of evaluate calls, one frame at a time, is used for the RNN, 
    # hidden state is preserved across evaluate calls
    # 
    # INPUT: A single frame. Implicitly this function also takes it's history as an input. 
    # OUTPUT: This function will return the encoded frame to be broadcast to the sender
    def evaluate(self, frame):
        pass



#Will implement later. 
# Probably wont be needed until we actually simulate on a real network
# Once we have the decoder the accuracy will be measured by the enviornment
class Receiver():
    def __init__(self):
        pass