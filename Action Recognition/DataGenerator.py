import numpy as np
import h5py
from tensorflow import keras
import random

class DataGenerator(keras.utils.Sequence):
    #Initialize
    def __init__(self, file_path, shape=(80,80), frames_per_sample=32, data_amount=300000, batch_size=32,offset=0):
        self.file_path=file_path
        #Assert that data_amount is <= available data
        with h5py.File(file_path,'r') as f:
            if data_amount > len(f["/frames/raw"]):
                self.data_amount = len(f["/frames/raw"])
            else:
                self.data_amount=data_amount
        #Instance Variables
        self.shape=shape
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.offset=offset
        self.skipped_frames=0
        self.buffered_frames = np.zeros((self.batch_size*self.frames_per_sample*50,)+self.shape)
        self.buffered_labels = np.zeros(self.batch_size*self.frames_per_sample*50)
        self.buffer_index = 0
        self.buffer_loc=0
        self.refill_buffer()

    def __len__(self):
        #How many batches in 1 epoch
        return int(np.floor((self.data_amount/self.frames_per_sample)/self.batch_size))

    def __getitem__(self, index):
        index*=self.batch_size*self.frames_per_sample+self.skipped_frames
        #Create a batch
        with h5py.File(self.file_path,'r') as f:
            if not self.frames_per_sample == 1:
                #Generate batch_size long clips of frames to input.
                return self.generate_chunk(index)
            else:
                #For when you only want 1 frame per sample in the batch (e.g. Conv2D)
                #Pick a random set of batch_size frames for the whole batch
                batch_data = np.zeros((self.batch_size,)+self.shape)
                batch_labels = np.zeros(self.batch_size)
                i=0
                while i<self.batch_size:
                    if f["/labels"][index]>=0:
                        batch_data[i]=f["/frames/raw"][index]
                        batch_labels[i]=f["/labels"][index]
                        i+=1
                    else:
                        self.skipped_frames+=1
                return batch_data[...,None],batch_labels


    def generate_chunk(self,batch_ind):
        chunk_data=np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        chunk_labels=np.zeros(self.batch_size)
        with h5py.File(self.file_path,'r') as f:
            i=0
            #Starting index for batch samples
            #index = batch_ind
            while i < self.batch_size:
                #Loop forward to add to the frame sequence
                cur_amount = 0
                while cur_amount < self.frames_per_sample:
                    if self.buffer_index>=len(self.buffered_frames):
                        self.buffer_loc+=1
                        self.refill_buffer()
                    label = self.buffered_labels[self.buffer_index]
                    if label >= 0:
                        chunk_data[i][cur_amount]=self.buffered_frames[self.buffer_index]
                        chunk_labels[i]=label
                        self.buffer_index+=1
                        cur_amount+=1
                    else:
                        self.buffer_index+=1
                i+=1
            #chunk_data = (chunk_data+np.random.normal(0,1,(self.batch_size,self.frames_per_sample,80,80)))
            #chunk_data[chunk_data<0]=0
            return chunk_data[...,None],chunk_labels
    def refill_buffer(self):
        load_len = len(self.buffered_frames)
        start = self.buffer_loc*load_len
        end = min((self.buffer_loc+1)*load_len,self.data_amount)
        with h5py.File(self.file_path,'r') as f:
            self.buffered_frames=f["/frames/raw"][start:end]
            self.buffered_labels=f["/labels"][start:end]
            if end >= self.data_amount:
                self.buffered_frames=buffered_frames[0:i-(i%(self.batch_size*self.frames_per_sample))]
                self.buffered_labels=buffered_labels[0:i-(i%(self.batch_size*self.frames_per_sample))]
            self.buffer_index=0
        
    def on_epoch_end(self):
        self.skipped_frames=0
        self.buffer_index=0
        self.buffer_loc=0
        self.buffered_frames = np.zeros((self.batch_size*self.frames_per_sample*50,)+self.shape)
        self.buffered_labels = np.zeros(self.batch_size*self.frames_per_sample*50)
        pass