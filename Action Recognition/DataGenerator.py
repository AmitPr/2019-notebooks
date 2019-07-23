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

    def __len__(self):
        #How many batches in 1 epoch
        return int(np.floor((self.data_amount/self.frames_per_sample)/self.batch_size/10))

    def __getitem__(self, index):
        index*=self.batch_size*self.frames_per_sample
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
                return batch_data[...,None],batch_labels


    def generate_chunk(self,batch_ind):
        chunk_data=np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        chunk_labels=np.zeros(self.batch_size)
        with h5py.File(self.file_path,'r') as f:
            i=0
            #Starting index for batch samples
            index = batch_ind
            while i < self.batch_size:
                #Loop forward to add to the frame sequence
                cur_amount = 0
                while cur_amount < self.frames_per_sample:
                    label = f["/labels"][index]
                    if label >= 0:
                        chunk_data[i][cur_amount]=f["/frames/raw"][index]
                        chunk_labels[i]=f["/labels"][index]
                        index+=1
                        cur_amount+=1
                    else:
                        index+=1
                i+=1
            #chunk_data = (chunk_data+np.random.normal(0,1,(self.batch_size,self.frames_per_sample,80,80)))
            #chunk_data[chunk_data<0]=0
            return chunk_data[...,None],chunk_labels
    
    def on_epoch_end(self):
        pass

    
#Reprocess a file for the data generator. -- used for old Data Generator
#This is done in advance so that python doesn't run out of memory when using multiprocessing.
def process_file(in_file_path, out_file_path,frames_per_sample=32, data_amount=300000,frame_shape=(80,80,1)):
    with h5py.File(in_file_path,'r') as f:
            #Load
            labels = (f['/labels'][:data_amount])
            data_amount=len(labels)
            combine_frames=(not frames_per_sample==1)
            if combine_frames:
                indices_to_copy=np.zeros(int(2*data_amount/frames_per_sample)+1,dtype=int)
                ind_i = 0
                #Array to copy blocks of data into
                temp_i = 0
                i = 0
                #label of continous set of matching data
                cur_label = labels[0]
                while i < data_amount:
                    #Find the end of the continous set of matching data
                    start = i
                    while labels[i]==cur_label:
                        i+=1
                        if i == data_amount:
                            break
                    end = i
                    #next set's label (if not at end)
                    if not i == len(labels):
                        cur_label=labels[i]
                    #Add block of continous data to new dataset
                    if end-start >= frames_per_sample and labels[start] >= 0:
                        indices_to_copy[ind_i]=start
                        indices_to_copy[ind_i+1]=end
                        ind_i+=2
                        labels[temp_i:temp_i+end-start]=labels[start:end]
                        temp_i+=end-start
                #trim to new size
                labels=labels[:temp_i]
                indices_to_copy=indices_to_copy[:ind_i]
                    
            with h5py.File(out_file_path,'w') as wf:
                wf.create_dataset("/frames",shape=((len(labels),)+frame_shape),dtype='float32')
                wf.create_dataset("/labels",data=labels)
                cur_i = 0
                for j in range(0,len(indices_to_copy),2):
                    start = indices_to_copy[j]
                    end = indices_to_copy[j+1]
                    wf["/frames"][cur_i:cur_i+(end-start)]=(f["/frames/raw"][start:end])[...,None]
                    cur_i+=end-start
                    print("Processing File: [",(j/len(indices_to_copy)),"%]",end='\r')
                #del data
                del labels