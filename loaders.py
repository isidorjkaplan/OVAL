import cv2
import numpy as np
import numpy
import torch

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
            return StopIteration()

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
            return StopIteration()

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


class VideoLoader():
    #Opens the file and initilizes the video
    def __init__(self, vid_id, filepath, batch_size, video_size=None):
        self.cap = cv2.VideoCapture(filepath)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.batch_size = batch_size
        self.video_size = video_size
        self.vid_id = vid_id
        self.num_frames_read = 0
        #print("Loading Video=%s with frames=%d and size=(%d,%d)" % (filepath, self.frameCount, self.frameWidth, self.frameHeight))

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_frames_read == self.frameCount:
            return StopIteration()

        buf = np.empty((self.batch_size, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        while (fc < self.batch_size and self.num_frames_read < self.frameCount  and ret):
            ret, buf[fc] = self.cap.read()
            if self.video_size is not None: #Optionally resize to specific size
                buf[fc] = cv2.resize(buf[fc], self.video_size, interpolation = cv2.INTER_LINEAR)
            fc += 1
            self.num_frames_read += 1

        if self.num_frames_read == self.frameCount:
            self.cap.release()

        buf = torch.FloatTensor(buf[:fc]).permute(0, 3, 1, 2)/255.0

        return buf, self.num_frames_read == self.frameCount

    def __del__(self):
        #Destroy all threads
        pass


class VideoDatasetLoader():
    #Opens the file and initilizes the video
    def __init__(self, directory, batch_size, video_size=None):
        self.last_video = 0
        self.video_directory = directory
        self.video_loaders = None
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def reset(self):
        if self.video_loaders is not None:
            del self.video_loaders
        self.video_loaders = [VideoLoader(vid_id, filepath, self.batch_size)  for vid_id,filepath in enumerate(glob.glob(os.path.join(self.video_directory, "*.mp4")))]
        print("Reset video loader dataset with %d Videos" % len(self.video_loaders))

    def __next__(self):
        if len(self.video_loaders) == 0:
            return StopIteration() #No videos left for this epoch, must reset

        loader_num = self.last_video % len(self.video_loaders)
        video_loader = self.video_loaders[loader_num]
        vid_id = video_loader.vid_id
        frames, done = next(video_loader)
        if done:
            del self.video_loaders[loader_num]
        
        return vid_id, frames
