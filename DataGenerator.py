import numpy as np
from PIL import Image
from tensorflow import keras 
class DataGenerator(keras.utils.Sequence):
    def __init__(self,img_data,batch_size=32):
        self.img_data=img_data
        self.batch_size=batch_size
        self.img_shape=img_data[0].shape
        return
    def __len__(self):
        return int(np.floor(len(self.img_data) / self.batch_size))
    def __getitem__(self,index):
        rotated = np.ndarray((self.batch_size,*self.img_shape))
        angles = np.ndarray(self.batch_size)
        for i in range(0,self.batch_size):
            rotated[i],angles[i]=self.__rotate_frame(self.img_data[index*self.batch_size+i])
        return rotated[...,None],angles
    def __rotate_frame(self, frame):
        a = np.random.random()*180
        return np.array(Image.fromarray(frame).rotate(a)),a