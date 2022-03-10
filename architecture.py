import copy
import multiprocessing
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import os

from torch.utils.tensorboard import SummaryWriter

#This class encompasses the sender. 
# DESC: Anything that is done at the sender is handled within this class
# INPUT: This class will take in frames, one at a time, representing the live video
#        Also takes in a pre-trained model from offline learning to be updated
# OUTPUT: This will output encoded frames, and periodically decoder models
class Sender():
    #Init function for Sender
    def __init__(self, autoencoder, reward_func, board, lr, max_buffer_size, update_threshold, loss_fn, enc_bytes, min_frames=10, live_device='cuda', train_device='cuda', fallback=None):
        #Live model is used for actively encoding frames, and stores the last broadcast model
        self.live_model = autoencoder.clone()
        #As we train with random encoding sizes we will keep track of a map enc_size->loss
        #This will measure the accuracy of each encoding size
        #We will use use an epslion-greedy stratagey between exploring different encoding sizes
        #And always using the best encoding size we see. Every itteration we update how it performed
        #When a particular encoding size does bad we try other ones more frequently 
        #For evaluation (not training) of the current frame we just use whatever encoding size has lowest loss
        #self.size_loss = [0 for _ in range(autoencoder.num_enc_layers)]
        #This is some reward function func(enc_size, loss) which we use to select the best encoding size
        self.reward_func = reward_func
        #Fallback is some trusted algorythem for encoding that we use if the error becomes intolerable
        #As it keeps training and ges better it will stop using the fallback
        self.fallback = fallback
        #How many frames we need to begin training. Can't start until our buffer is large enough
        self.min_frames = min_frames
        #How many frames can the largest our buffer be. If it is larger we start throwing out old frames
        self.max_buffer_size = max_buffer_size
        self.loss_fn = loss_fn

        self.eval_hidden = None

        self.board = board
        self.live_device = live_device
        self.train_device = train_device

        self.train_q = multiprocessing.Queue()
        self.model_q = multiprocessing.Queue()

        self.enc_bytes = enc_bytes

        self.update_threshold = update_threshold
        self.lr = lr
        pass

    # PUBLIC METHODS

    def init_train(self):
        self.iter = 0
        #LOCAL VARIABLES
        self.buffer = []
        #The hidden state of our LSTM. Will carry over even as we switch active models. Used for the real-time frames
        self.train_hidden = [None, None]
        #Train model is the one that we are actively training. Periodicailly we set live_model = train_model with a broadcast
        if self.update_threshold is not None:
            self.train_model = self.live_model.clone()
            self.live_hidden = [None,None]
        else:
            self.train_model = self.live_model #only one model
        params = list(self.train_model.encoder.parameters()) +  list(self.train_model.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    
    #Run one iteration of training on its local buffer to train AE
    # Does not modify the hidden state of the LSTM used for live decoding
    # This function performs a step and does an update
    #
    # INPUT: Nothing. Trains on it's history of frames from evaluate calls. 
    #        It will also optionally take a tensorboard. It will print training related statistics to it
    # OUTPUT: Either returns "None" or a "Decoder" to be broadcast on the network. 
    # INTERNAL: Updates the self.size_loss for encoding size used this iter. 
    def step(self, board=None):
        #Cant train on an empty buffer
        while not self.train_q.empty():
            frame = self.train_q.get()
            self.buffer.append(frame)
        while len(self.buffer) > self.max_buffer_size:
            del self.buffer[0]

        if len(self.buffer) < self.min_frames:
            time.sleep(0)#Yield the thread
            return None

        #del data
        torch.cuda.empty_cache()

        start = time.time()
            
        data = torch.cat(self.buffer, dim=0).detach().to(self.train_device) #Construct the training data


        self.train_model.to(self.train_device)
        dec_frame, self.train_hidden[0] = self.train_model.encoder(data, self.train_hidden[0])
        dec_frame, self.train_hidden[1] = self.train_model.decoder(dec_frame, self.train_hidden[1])
        #Truncate to proper size
        data_mse = data[:,:,:dec_frame.shape[2], :dec_frame.shape[3]] #Due to conv fringing, not same size. Almost same size. Just cut
        dec_frame = dec_frame[:,:,:data_mse.shape[2], :data_mse.shape[3]]
        
        #print("%s -> %s" % (str(data.shape), str(data_mse.shape)))
        loss_train = self.loss_fn(dec_frame, data_mse) #Compute the loss
        self.board.put(("sender/loss_train (batch)", loss_train.detach().cpu().item(), self.iter))

        if self.update_threshold is not None:
            self.live_model.to(self.train_device)
            x, self.live_hidden[0] = self.live_model.encoder(data, self.live_hidden[0])
            x, self.live_hidden[1] = self.live_model.decoder(x, self.live_hidden[1])
            loss_live = self.loss_fn(x, data_mse) #Compute the loss
            self.board.put(("sender/loss_live (batch)", loss_live.detach().cpu().item(), self.iter))
            
            rel_err = (loss_live/loss_train - 1).detach().cpu().item()
            self.board.put(("sender/relative_error (batch)", rel_err, self.iter))

        loss_train.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_model.save_model()
        
        self.board.put(("timing/train_iter (sec)", time.time() - start, self.iter))

        #del data
        torch.cuda.empty_cache()

        #Evaluate how live model is doing

        self.iter+=1
        #FOR NOW ALWAYS UPDATE, CHANGE THIS LATER
        if self.update_threshold is None or rel_err >= self.update_threshold: #Should update in 5% difference
            if self.update_threshold is not None:
                print("Broadcasting Model Update")
            #Send to the thread handling evaluation
            self.model_q.put(self.train_model.encoder.cpu().state_dict())
            #Update for should_update evaluation
            del self.live_model
            self.live_model = self.train_model.clone()  
            return self.train_model.decoder.cpu().state_dict()

        return None
        pass
    
    #Evaluate a single frame. 
    # This will use previous evaluate call hidden state for RNN
    # Each sequence of evaluate calls, one frame at a time, is used for the RNN, 
    # hidden state is preserved across evaluate calls
    # 
    # INPUT: A single frame. Implicitly this function also takes it's history as an input. 
    # OUTPUT: This function will return the encoded frame to be broadcast to the sender
    #         Note the encodign size used will be determined by self.size_loss and self.reward_func each call
    # INTERNAL: This will update "self.hidden" for the next call
    def evaluate(self, frame):
        #Save value onto our training buffer
        #frame.share_memory_()
        self.live_model.to(self.live_device)
        if not self.model_q.empty():
            self.live_model.encoder.load_state_dict(self.model_q.get())
            self.live_model.to(self.live_device)
        #Actually run the encoder on the frame
        enc_state, self.eval_hidden = self.live_model.encoder(frame.to(self.live_device), self.eval_hidden)
        self.train_q.put(frame)
        enc_state = enc_state.cpu()
        torch.cuda.empty_cache()

        return enc_state.type(self.enc_bytes)



#Will implement later. 
# Probably wont be needed until we actually simulate on a real network
# Once we have the decoder the accuracy will be measured by the enviornment
class Receiver():
    def __init__(self):
        pass