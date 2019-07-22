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
                    index = random.randint(self.offset,self.data_amount+self.offset-2)
                    if f["/labels"][index]>=0:
                        batch_data[i]=f["/frames/raw"][index]
                        batch_labels[i]=f["/labels"][index]
                        i+=1
                return batch_data[...,None],batch_labels


    def generate_chunk(self,batch_ind):
        chunk_data=np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        chunk_labels=np.zeros(self.batch_size)
        cur_amount = 0
        with h5py.File(self.file_path,'r') as f:
            for i in range(0,self.batch_size):
                #Starting index for each sample
                index = self.frames_per_sample*random.randint(0,self.data_amount/self.frames_per_sample)
                #alternative index: random.randint(self.offset,self.data_amount+self.offset-(self.frames_per_sample*2))
                #Loop forward to add to the frame sequence
                while cur_amount < self.frames_per_sample:
                    label = f["/labels"][index]
                    if label >= 0:
                        chunk_data[i][cur_amount]=f["/frames/raw"][index]
                        index+=1
                        cur_amount+=1
                    else:
                        index+=1
                #Add end of sequence frame's label to label list
                chunk_labels[i]=f["/labels"][index]
            return chunk_data[...,None],chunk_labels
    
    def on_epoch_end(self):
        pass

    
#Reprocess a file for the data generator. -- used for old Data Generator
#This is done in advance so that python doesn't run out of memory when using multiprocessing.
def process_file(in_file_path, out_file_path,frames_per_sample=32, data_amount=300000,frame_shape=(80,80,1)):
    with h5py.File(in_file_path,'r') as f:
            #Load
            labels = (f['/labels'][:data_amount])
            #data = f['/frames/raw'][:data_amount]
            #Change dimensions for networks
            #data = (data[(labels>=0)])[...,None]
            #labels = labels[(labels>=0)]
            data_amount=len(labels)
            combine_frames=(not frames_per_sample==1)
            if combine_frames:
                indices_to_copy=np.zeros(int(2*data_amount/frames_per_sample)+1,dtype=int)
                ind_i = 0
                #Array to copy blocks of data into
                #temp_data = np.zeros(data.shape)
                #current index in temp_data
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
                        #temp_data[temp_i:temp_i+end-start]=data[start:end]
                        indices_to_copy[ind_i]=start
                        indices_to_copy[ind_i+1]=end
                        ind_i+=2
                        labels[temp_i:temp_i+end-start]=labels[start:end]
                        temp_i+=end-start
                #trim to new size
                #temp_data=temp_data[:temp_i]
                labels=labels[:temp_i]
                indices_to_copy=indices_to_copy[:ind_i]
                #data=temp_data
                    
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