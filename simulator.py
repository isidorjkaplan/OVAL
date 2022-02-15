import cv2
import numpy
import torch
import time
import torch.nn.functional as F
import copy
import numpy as np
from multiprocessing import Process, Value
from torch.multiprocessing import Queue
import cv2
import matplotlib.pyplot as plt
import ctypes
import traceback
from skvideo.io import FFmpegWriter
import ffmpeg


from torch.utils.tensorboard import SummaryWriter

# Fixes "too many files open" errors
torch.multiprocessing.set_sharing_strategy("file_system")

# This enviornment is used for the tests without federated learning. We are just monitoring one sender's ability to adapt
# We will run this on many seperate videos and collect the average for test reporting in the paper
# In this setup, we don't actually use a receiver. The sender is a black box which will take in frames we feed it 
# We will keep track of any information it broadcasts for the sake of monitoring the information
# Any encoded frames it sends us, we will use it's last decoder to decode it and measure it's accuracy
# The sender thinks it is sending to a real network with real decoders, but we intercept and simply calculate its test performance here
# In future tests, we use the same sender without modification, but instead we will have many and maintain the federated model in env
class SingleSenderSimulator():
    def __init__(self, sender, board, server=None, live_video=False):
        #Sender does all the heavy lifting on training, we are just an interface for sender and the real world and testing
        self.sender = sender
        #A tensorboard which we will plot statitsics about the accuracy and all that of our sender
        self.board = board
        #self.writer = SummaryWriter(board)
        #At the beginning server=None. That is to say, we won't actually broadcast. 
        #We will just discard the messages once we are done without sending them anywhere, just look at them for testing evaluation
        #Later on we will modify this to support a "server" where we will actually forward the broadcasts
        self.server = server
        self.live_video = live_video
        #Live decoder, we keep a copy here in the simulator since we dont know what is happening internal to sender
        self.decoder = sender.live_model.clone().decoder
        self.done = Value(ctypes.c_bool)
        self.done.value = False
        self.data_q = Queue()
        self.model_q = Queue()
        pass
    
    #Start the entire process, starts both train and video thread, runs until video is complete, and then terminates
    # When this returns it must have killed both the train and video thread
    # Will return some final statistics such as the overall error rate, overall network traffic, overall accuracy for the entire video
    def start(self, video, runtime, out_file, rate, batch_size, loss_fn):
        p_train = Process(target=self.train_thread)
        p_train.start() #Start training and then go to the live video feed

        p_recv = Process(target=self.receive_thread, args=(out_file,rate,batch_size,loss_fn))
        p_recv.start()
        try:
            self.video_thread(video, runtime)
            p_train.join() #Wait for training thread to kill itself safely
            p_recv.join()
        except KeyboardInterrupt as e:
            print("Captured Keyboard Interrupt")
            print(traceback.format_exc())
            p_train.kill()
            p_recv.kill()
            p_train.join()
            p_recv.join()
        except Exception as e:
            print(traceback.format_exc())
            p_train.kill()
            p_recv.kill()
            p_train.join()
            p_recv.join()

            
        pass #Return
    
    # Manages the training loop for the sender, runs continiously
    # Any network updates it sends here we will keep track of
    def train_thread(self):
        self.sender.init_train()
        while not self.done.value:
            params = self.sender.step()
            if params is not None:
                self.model_q.put(params)
            pass
        print("Train Thread Terminated")

    #The critical path. Simply steps through the frames one by one with an appropriate delay to simulate real-time input
    #For each frame it takes it, has the sender encode it
    #For testing this will then evaluate the accuracy of the encoding and keep track of network traffic
    #It's network traffic, and evaluation of the encoded videos sent will be tested here and plotted
    def video_thread(self, video, runtime):
        start = time.time()
        if runtime is not None:
            stop_time = start + runtime
        frame_num = 0
        for i,frame in enumerate(video):
            if frame is None:
                break
            if runtime is not None and time.time() > stop_time:
                break
            #Perform encoding and transmit it
            encoded = self.sender.evaluate(frame).detach()
            print(f"reading frame: {frame_num}")
            frame_num+=1
            #encoded.share_memory_()
            #frame.share_memory_()
            self.data_q.put((encoded, frame))
            now = time.time()
            if abs(now-start)>0.00001: #Somehow I was getting a divide by zero error?
                self.board.put(("timing/send_fps (frames/sec)", 1/(now - start), i))
            start = time.time()            #Evaluate the error on our encoding to report for testing set
        self.done.value = True # kills the train thread
        self.data_q.put(None) #Signify it is done
        print("Finished reading Video")
            
    def receive_thread(self, out_file, rate, batch_size, loss_fn):
        frame_num = 0
        type_sizes = {torch.float16:2, torch.float32:4, torch.float64:8}
        out = None
        if out_file is not None:
            out = FFmpegWriter(out_file)
        batch_decoded_np = []
        batch_count = 0
        while True:
            num_bytes = 0
            #THIS IS TOO SLOW. Must do this in another thread
            if not self.model_q.empty():
                num_bytes += sum((p.numel()*type_sizes[p.dtype]) for p in self.decoder.parameters())
                self.decoder.load_state_dict(self.model_q.get())
            #This is done here instead of send thread to avoid delaying critical path measurements
            data = self.data_q.get()
            if data is None:
                print("Video Stream Terminated")
                break
            encoded, frame = data

            num_bytes += encoded.numel()*type_sizes[encoded.dtype] #Check what type was used on network
            encoded = encoded.type(frame.dtype) #We can now upscale its type back to 32 bit for evaluation
            dec_frame = self.decoder(encoded).detach()
            frame = frame[:,:,:dec_frame.shape[2], :dec_frame.shape[3]] #Due to conv fringing, not same size. Almost same size. Just cut
            uncomp_bytes = frame.shape[1]*frame.shape[2]*frame.shape[3]*1 #For uncompressed, 1 byte per channel * C*L*W is total size
            #frame = self.video.get_frame(frame_num)
            error = F.mse_loss(frame, dec_frame).detach() #Temporary, switch later     
            #print("loss_v=%g" % error)
            self.board.put(("receiver/realtime frame loss", error, frame_num)) 
            self.board.put(("receiver/compression factor (original/compressed)", uncomp_bytes/num_bytes, frame_num))
            #Live display of video 
            dec_np_frame = dec_frame[0].permute(2, 1, 0).numpy()
            dec_np_frame = np.uint8(255*dec_np_frame)
            batch_count+=1
            batch_decoded_np.append(dec_np_frame)
            if out_file is not None and batch_count == batch_size:
                for i in range(batch_size):
                    out.writeFrame(batch_decoded_np[i])
                batch_decoded_np = []
                batch_count=0
                print(f"writing frame: {frame_num}")
            if self.live_video and frame_num % 30 == 0:
                cv2.imshow("Decoded", dec_np_frame)
                cv2.imshow("Real", frame[0].permute(2, 1, 0).numpy())
                cv2.waitKey(1)
            frame_num+=1
            #plt.imshow(dec_frame[0].permute(1, 2, 0))
            #plt.show(block=False)
            #print("Received encoded frame with loss = " + str(error.item()))
        if out_file is not None:
            print("outfile closed")
            out.close()
        print("Receive thread terminated")
        cv2.destroyAllWindows()
        pass
    #PRIVATE FUNCTIONS

#Simulates a file as being a live video stream returning rate frames per second
class CameraVideoSimulator():
    #Opens the file and initilizes the video
    def __init__(self, rate=30, size=None):

        #Parameters for frame reading
        self.num_frames_read = 0
        self.last_frame_time = time.time()
        self.time_between_frames = 1.0/rate

        self.stream = cv2.VideoCapture(0)
        self.frameWidth = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))


    def __iter__(self):
        return self

    def __next__(self):
        return self.next_frame()


    
    def next_frame(self):
        ret, frame = self.stream.read()
        if not ret:#Check for error
            return None

        frame = torch.FloatTensor(frame).permute(2, 1, 0)/255.0
        frame = frame.view(1, 3, self.frameWidth, self.frameHeight)

        #Sleep so that we ensure appropriate frame rate, only return at the proper time
        now = time.time()
        sleep_time = self.time_between_frames - (now - self.last_frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        #If negative this should have arrived already and we are behind so just go so time gets earlier
        self.last_frame_time = now + sleep_time
        #Return value
        return frame

    def __del__(self):
        self.stream.release()


#Simulates a file as being a live video stream returning rate frames per second
class VideoSimulator():
    #Opens the file and initilizes the video
    def __init__(self, filepath, rate=30, size=None, repeat=False):
        cap = cv2.VideoCapture(filepath)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Loading Video=%s with frames=%d and size=(%d,%d)" % (filepath, self.frameCount, self.frameWidth, self.frameHeight))
        #Parameters for frame reading
        self.num_frames_read = 0
        self.last_frame_time = time.time()
        self.time_between_frames = 1.0/rate
        self.repeat = repeat

        buf = np.empty((self.frameCount, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < self.frameCount  and ret):
            ret, buf[fc] = cap.read()
            if size is not None: #Optionally resize to specific size
                buf[fc] = cv2.resize(buf[fc], size, interpolation = cv2.INTER_LINEAR)
            fc += 1
        cap.release()

        self.buffer = buf #save buffer
        #self.frames = torch.FloatTensor(buf)/255
        #self.frames = self.frames.permute(0, 3, 1, 2)#Make channel major

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_frame()

    def get_frame(self, i):
        assert self.repeat or i < self.frameCount
        frame = torch.FloatTensor(self.buffer[i % self.frameCount]).permute(2, 1, 0)/255.0
        frame = frame.view(1, 3, self.frameWidth, self.frameHeight)
        return frame

    
    def next_frame(self):
        #Do all the reading and processing of the frame
        if self.num_frames_read >= self.frameCount and not self.repeat:
            return None

        frame = self.get_frame(self.num_frames_read)
        self.num_frames_read+=1
        #Sleep so that we ensure appropriate frame rate, only return at the proper time
        now = time.time()
        sleep_time = self.time_between_frames - (now - self.last_frame_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        #If negative this should have arrived already and we are behind so just go so time gets earlier
        self.last_frame_time = now + sleep_time
        #Return value
        return frame



#There will be one server. It will take input from each client and then broadcast it to each other client
#When we incorperate federated learning this will also serve as the federated learning server
#Later on the "server" parameter of sender will be an IP address to wherever this is hosted
class RealSimulatorServer():
    def __init__(self, senders):
        pass