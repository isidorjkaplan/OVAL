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
    def video_thread():
        pass

    #Private function called by both train and video thread to keep track of broadcast data
    def __record_network_traffic(num_bytes):
        pass


class FederatedSimulator():
    def __init__(self, senders):
        pass