# This enviornment is used for the tests without federated learning. We are just monitoring one sender's ability to adapt
# We will run this on many seperate videos and collect the average for test reporting in the paper
# In this setup, we don't actually use a reciever. The sender is a black box which will take in frames we feed it 
# We will keep track of any information it broadcasts for the sake of monitoring the information
# Any encoded frames it sends us, we will use it's last decoder to decode it and measure it's accuracy
# The sender thinks it is sending to a real network with real decoders, but we intercept and simply calculate its test performance here
# In future tests, we use the same sender without modification, but instead we will have many and maintain the federated model in env
class SingleSenderSimulator():
    def __init__(self, sender):
        pass
    
    #Start the entire process, starts both train and video thread and then returns
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

    #Start and setup Tensorboard, responsibility of the simulator to print statistics
    def __start_tensorboard():
        pass

    #Private function called by both train and video thread to keep track of broadcast data
    #Plots the data on the tensorboard so we can monitor the data transmission
    def __record_network_traffic(num_bytes):
        pass


# REAL NETWORK TESTS
#This is for use later. What it will do is that several different actual users to test on a real network

#Each client runs on a seperate machine
class RealSimulatorClient():
    def __init__(self, senders):
        pass

    def start(runtime):
        pass

    # Manages the training loop for the sender, runs continiously
    # Any network updates it sends here we will keep track of
    def train_thread():
        pass

    #Feeds in the frames at a fixed FPS rate and then sends the results onto the network to the server
    def video_sender_thread():
        pass

    #Video reciever thread, recieves input here and decodes it
    def video_reciever_thread():
        pass

#There will be one server. It will take input from each client and then broadcast it to each other client
#When we incorperate federated learning this will also serve as the federated learning server
class RealSimulatorServer():
    def __init__(self, senders):
        pass

    #TODO