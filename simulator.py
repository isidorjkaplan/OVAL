# This enviornment is used for the tests without federated learning. We are just monitoring one sender's ability to adapt
# We will run this on many seperate videos and collect the average for test reporting in the paper
# In this setup, we don't actually use a reciever. The sender is a black box which will take in frames we feed it 
# We will keep track of any information it broadcasts for the sake of monitoring the information
# Any encoded frames it sends us, we will use it's last decoder to decode it and measure it's accuracy
# The sender thinks it is sending to a real network with real decoders, but we intercept and simply calculate its test performance here
# In future tests, we use the same sender without modification, but instead we will have many and maintain the federated model in env
class SingleSenderSimulator():
    def __init__(self, sender, board, server=None):
        #Sender does all the heavy lifting on training, we are just an interface for sender and the real world and testing
        self.sender = sender
        #A tensorboard which we will plot statitsics about the accuracy and all that of our sender
        self.board = board
        #At the beginning server=None. That is to say, we won't actually broadcast. 
        #We will just discard the messages once we are done without sending them anywhere, just look at them for testing evaluation
        #Later on we will modify this to support a "server" where we will actually forward the broadcasts
        self.server = server
        pass
    
    #Start the entire process, starts both train and video thread, runs until video is complete, and then terminates
    # When this returns it must have killed both the train and video thread
    # Will return some final statistics such as the overall error rate, overall network traffic, overall accuracy for the entire video
    def start(runtime):
        pass
    
    # Manages the training loop for the sender, runs continiously
    # Any network updates it sends here we will keep track of
    def train_thread():
        pass

    #The critical path. Simply steps through the frames one by one with an appropriate delay to simulate real-time input
    #For each frame it takes it, has the sender encode it
    #For testing this will then evaluate the accuracy of the encoding and keep track of network traffic
    #It's network traffic, and evaluation of the encoded videos sent will be tested here and plotted
    def video_thread():
        pass

    #PRIVATE FUNCTIONS

    #Private function called by both train and video thread to keep track of broadcast data
    #Plots the data on the tensorboard so we can monitor the data transmission
    def __record_network_traffic(num_bytes):
        pass


#There will be one server. It will take input from each client and then broadcast it to each other client
#When we incorperate federated learning this will also serve as the federated learning server
#Later on the "server" parameter of sender will be an IP address to wherever this is hosted
class RealSimulatorServer():
    def __init__(self, senders):
        pass