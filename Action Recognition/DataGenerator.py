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
                 labels_structured=False,
                 sliding_window=0,
                 label_range=(0,54),
                 standardize=False):
        
        self.file_path=file_path
        #Assert that data_amount is <= available data
        with h5py.File(file_path,'r') as f:
            if data_amount > len(f["/frames/raw"]):
                self.data_amount = len(f["/frames/raw"])
                print("More data than available")
            else:
                self.data_amount=data_amount
            f.close()
        #Instance Variables
        self.shape=shape
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample
        self.offset=offset
        self.skipped_frames=0
        self.labels_structured = labels_structured
        self.standardize=standardize
        self.label_range=label_range
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
            f.close()
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
        return self.generate_chunk(index)

    def generate_chunk(self,batch_ind):
        if self.frames_per_sample > 1:
            chunk_data=np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        else:
            chunk_data=np.zeros((self.batch_size,)+self.shape)
        if self.labels_structured:
            aux_labels=np.zeros((self.batch_size,self.frames_per_sample-1))
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
            #to_remove = np.setxor1d(uniques[0],np.arange(self.label_range[0],self.label_range[1]))
            #for x in to_remove:
            #    if x in uniques[0]:
            #        amount_to_remove = uniques[1][np.where(unique==x)[0]]
            #        end+=amount_to_remove
            #preload_labels=f["/labels"][start:end]
            if -5 in uniques[0]:
                negatives = uniques[1][0]
                end += negatives+20
                preload_labels = f["/labels"][start:end]
            preload = f["/frames/raw"][start:end].astype('float32')
            f.close()
        
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
                    frame = preload[buffer_index]
                    if self.standardize:
                        #Standardize Data
                        frame-=np.mean(frame)
                        frame/=np.std(frame)
                    if self.frames_per_sample > 1:
                        chunk_data[i][cur_amount]=frame
                    else:
                        chunk_data[cur_amount]=frame
                    if self.labels_structured and cur_amount < self.frames_per_sample-1:
                        aux_labels[i][cur_amount]=label
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
        if self.labels_structured:
            return [chunk_data[...,None],aux_labels[...,None]],chunk_labels
        return chunk_data[...,None],chunk_labels
    
    def on_epoch_end(self):
        self.skipped_frames=0
        pass
