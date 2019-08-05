import numpy as np
import h5py
from tensorflow import keras
import random

class DataGenerator(keras.utils.Sequence):
    #Initialize
    def __init__(self,
                 file_path,
                 shape=(80,80),
                 frames_per_sample=32,
                 data_amount=300000,
                 batch_size=32,
                 offset=0,
                 use_crf=False,
                 sliding_window=0):
        
        self.file_path=file_path
        #Assert that data_amount is <= available data
        with h5py.File(file_path,'r') as f:
            if data_amount > len(f["/frames/raw"]):
                self.data_amount = len(f["/frames/raw"])
                print("More data than available")
            else:
                self.data_amount=data_amount
        #Instance Variables
        self.shape=shape
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.offset=offset
        self.skipped_frames=0
        self.use_crf = use_crf
        #Sliding Window setup
        if sliding_window == 0:
            self.window=False
        else:
            self.window=True
            self.slide_amt=sliding_window

    def __len__(self):
        #How many batches in 1 epoch
        with h5py.File(self.file_path,'r') as f:
            negatives = 0
            uniques = np.unique(f["/labels"][0:self.data_amount],return_counts=True)
            if uniques[0][0]<0:
                negatives+=uniques[1][0]
            if self.window:
                return int(np.floor(((self.data_amount-negatives-self.frames_per_sample+1)/self.batch_size*self.slide_amt)-1))
                #int(np.floor((self.data_amount-negatives)/((self.batch_size-1)*self.slide_amt+self.frames_per_sample)))
            else:
                return int(np.floor((self.data_amount/self.frames_per_sample)/self.batch_size/2)) - negatives

    def __getitem__(self, index):
        if not self.window or self.frames_per_sample == 1:
            index*=self.batch_size*self.frames_per_sample
            index+=self.skipped_frames
        else:
            index*=self.batch_size*self.slide_amt
            index+=self.skipped_frames
        #Create a batch
        with h5py.File(self.file_path,'r') as f:
            if not self.frames_per_sample == 1:
                #Generate batch_size long clips of frames to input.
                return self.generate_chunk(index)
            else:
                #For when you only want 1 frame per sample in the batch (e.g. Conv2D)
                #Pick a random set of batch_size frames for the whole batch
                chunk_data=np.zeros((self.batch_size,)+self.shape)
                chunk_labels=np.zeros(self.batch_size)
                with h5py.File(self.file_path,'r') as f:
                    i=0
                    #Starting index for batch samples
                    #Preload frames for the batch to reduce disk time
                    preload = f["/frames/raw"][index+self.skipped_frames+self.offset:index+self.batch_size+self.skipped_frames+self.offset]
                    while i < self.batch_size:
                        label = f["/labels"][index+self.offset]
                        if label >= 0:
                            #Get index in preload array
                            buffer_index = i+self.skipped_frames
                            #If we have skipped frames this batch then the preload will run out and we will have to read from disk
                            #Only reads the same amount as the number of skipped frames.
                            if buffer_index >= len(preload):
                                chunk_data[i]=f["/frames/raw"][index+self.offset]
                            #Otherwise just take from the preload array.
                            else:
                                chunk_data[i]=preload[buffer_index]
                            chunk_labels[i]=label
                            index+=1
                        else:
                            self.skipped_frames+=1
                            index+=1
                        i+=1
                    return chunk_data[...,None],chunk_labels


    def generate_chunk(self,batch_ind):
        chunk_data=np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        if self.use_crf:
            chunk_labels=np.zeros(self.batch_size*self.frames_per_sample)
        else:
            chunk_labels=np.zeros(self.batch_size)
        with h5py.File(self.file_path,'r') as f:
            i=0
            #Starting index for batch samples
            index = batch_ind
            #Preload frames for the batch to reduce disk time
            if not self.window:
                start = (index*self.frames_per_sample*self.batch_size)+self.skipped_frames+self.offset
            else:
                start = index + self.offset
            end = start + self.frames_per_sample*self.batch_size
            preload_labels = f["/labels"][start:end]
            uniques = np.unique(preload_labels,return_counts=True)
            if -5 in uniques[0]:
                negatives = uniques[1][0]
                end += negatives+20
                preload_labels = f["/labels"][start:end]
            preload = f["/frames/raw"][start:end]
        
        buffer_index = 0
        while i < self.batch_size:
            #Loop forward to add to the frame sequence
            cur_amount = 0
            while cur_amount < self.frames_per_sample:
                #Get index in preload array
                label = preload_labels[buffer_index]
                #If the label isn't invalid (-5)
                if label >= 0:
                    #Add data to batch data, label to batch labels
                    chunk_data[i][cur_amount]=preload[buffer_index]
                    if self.use_crf:
                        chunk_labels[i+cur_amount]=label
                    else:
                        chunk_labels[i]=label
                    index+=1
                    cur_amount+=1
                else:
                    self.skipped_frames+=1
                    index+=1
                buffer_index+=1
            i+=1
            #Sliding window, reset index to first frame + slide_amt of last sample
            if self.window:
                batch_ind += self.slide_amt
                index = batch_ind
        return chunk_data[...,None],chunk_labels
    
    def on_epoch_end(self):
        self.skipped_frames=0
        pass
