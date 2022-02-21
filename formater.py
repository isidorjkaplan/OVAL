import cv2
import numpy as np

#class to handle video format conversion
class VideoFormatConverter():
    def __init__(self, output_format = "bgr"):
        #initialize class by stating which output format you want, either "gbr or hsv for now"
        #assume input is always in bgr format
        self.output = output_format
    
    def encode(self, input_array):
        if (isinstance(input_array, np.ndarray)): #ensure that input is a numpy ndarray
            if (len(input_array.shape) == 4): #process batched input
                return self.batch_encode(input_array)
            elif (len(input_array.shape) == 3): #process single image input
                return self.single_encode(input_array)
            else: #raise value error if improper array passed
                raise ValueError("INPUT ARRAY MUST BE SINGLE IMAGE FOR BATCH OF IMAGES")
        else: #raise value error if numpy ndarray not passed
            raise ValueError("MUST PASS IN NDARRAY")
    
    def batch_encode(self, input_array):
        if self.output == "bgr": #convert to bgr not necessary as it should already be bgr
            return input_array
        elif self.output == "hsv":
            for i in range(len(input_array)):
                input_array[i] = cv2.cvtColor(input_array[i], cv2.COLOR_BGR2HSV)
            return input_array
        else: #invalid output type, raise value error
            raise ValueError("INVALID OUTPUT TYPE")
    
    def single_encode(self, input_array):
        if self.output == "bgr": #convert to bgr
            return self.encode_to_bgr(input_array)
        elif self.output == "hsv": #convert to hsv
            return self.encode_to_hsv(input_array)
        else: #raise value error invalid output type
            raise ValueError("INVALID OUTPUT TYPE")
    
    def encode_to_bgr(self, input_array):
        return input_array #just return itself because the input is assumed to bgr
    
    def encode_to_hsv(self, input_array):
        return cv2.cvtColor(input_array, cv2.COLOR_BGR2HSV) #convert image to hsv then return

    def decode(self, output_array):
        if (isinstance(output_array, np.ndarray)): #ensure that input is a numpy ndarray
            if (len(output_array.shape) == 4): #process batched input
                return self.decode_batch(output_array)
            elif (len(output_array.shape) == 3): #process single image input
                return self.decode_single(output_array)
            else: #raise value error if improper array passed
                raise ValueError("INPUT ARRAY MUST BE SINGLE IMAGE FOR BATCH OF IMAGES")
        else: #raise value error if numpy ndarray not passed
            raise ValueError("MUST PASS IN NDARRAY")
    
    def decode_batch(self, output_array):
        if self.output == "bgr": #convert to bgr not necessary as it should already be bgr
            return output_array
        elif self.output == "hsv":
            for i in range(len(output_array)):
                output_array[i] = cv2.cvtColor(output_array[i], cv2.COLOR_HSV2BGR)
            return output_array
        else: #invalid output type, raise value error
            raise ValueError("INVALID OUTPUT TYPE")
    
    def decode_single(self, output_array):
        if (self.output == 'bgr'):
            return output_array
        elif (self.output == "hsv"):
            return cv2.cvtColor(output_array, cv2.COLOR_HSV2BGR)
        else:
            raise ValueError("INVALID COLOR TYPE")
    
if (__name__ == "__main__"):
    formater = VideoFormatConverter(output_format='hsv')
    img = cv2.imread("test/Test_picture.jpg")
    print(img.shape)
    img = formater.encode(img)
    print(img.shape)
    img = formater.decode(img)
    print(img.shape)
    cv2.imwrite('test/new_test.jpg', img)