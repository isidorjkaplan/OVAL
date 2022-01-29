#Simulate the enviornment and measure statistics
class Environment():
    def __init__(self, sender):
        pass
    
    #Start the entire process, starts both train and video thread and then returns
    def start(runtime):
        pass
    
    #Manages the training loop for the sender, runs continiously
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


    