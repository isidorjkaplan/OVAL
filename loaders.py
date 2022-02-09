import cv2
import numpy as np
import numpy
import torch
import glob, os

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
    def __init__(self, filepath, rate=30, video_size=None, repeat=False):
        self.last_frame_time = time.time()
        self.time_between_frames = 1.0/rate
        self.repeat = repeat
        self.filepath = filepath
        self.video_size = video_size
        self.video_loader = VideoLoader(self.filepath, 1, video_size=self.video_size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_frame()

    def next_frame(self):
        #Do all the reading and processing of the frame
        if self.num_frames_read >= self.frameCount and not self.repeat:
            return None

        frame = next(self.video_loader)
        if frame is None: #Restart video loader
            if not self.repeat:
                return None
            print("VideoSimulator: Loader is over. Restarting new video loader.")
            del self.video_loader
            self.video_loader = VideoLoader(self.filepath, 1, video_size=self.video_size)
            frame = next(self.video_loader)


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
    def __init__(self, filepath, batch_size, video_id=None, video_size=None, max_frames=None):
        self.cap = cv2.VideoCapture(filepath)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            self.frameCount = min(self.frameCount, max_frames)
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.batch_size = batch_size
        self.video_size = video_size
        self.vid_id = video_id
        self.max_frames = None
        self.num_frames_read = 0
        #print("Loading Video=%s with frames=%d and size=(%d,%d)" % (filepath, self.frameCount, self.frameWidth, self.frameHeight))

    def __iter__(self):
        return self

    def __next__(self):
        if self.num_frames_read == self.frameCount:
            return None

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
    def __init__(self, directory, batch_size, video_size=None, max_frames=None):
        self.last_video = 0
        self.video_directory = directory
        self.video_loaders = None
        self.batch_size = batch_size
        self.max_frames = None

    def __iter__(self):
        return self

    def reset(self):
        if self.video_loaders is not None:
            del self.video_loaders
        self.video_loaders = [VideoLoader(filepath, self.batch_size, video_id=vid_id, max_frames=self.max_frames)  for vid_id,filepath in enumerate(glob.glob(os.path.join(self.video_directory, "*.mp4")))]
        print("Reset video loader dataset with %d Videos" % len(self.video_loaders))

    def __next__(self):
        if len(self.video_loaders) == 0:
            return None #No videos left for this epoch, must reset

        loader_num = self.last_video % len(self.video_loaders)
        video_loader = self.video_loaders[loader_num]
        vid_id = video_loader.vid_id
        frames, done = next(video_loader)
        if done:
            del self.video_loaders[loader_num]
        
        return vid_id, frames
