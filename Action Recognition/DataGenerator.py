import numpy as np
import h5py
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    #Initialize
    def __init__(self, file_path, shape=(80,80,1), frames_per_sample=32, data_amount=300000, batch_size=32):
        self.file_path=file_path
        with h5py.File(file_path,'r') as f:
            self.data_amount = len(f["/frames"])
        
        self.shape=shape
        self.batch_size = batch_size
        self.frames_per_sample = frames_per_sample

    def __len__(self):
        #How many batches in 1 epoch
        return int(np.floor(self.data_amount/self.frames_per_sample/self.batch_size))

    def __getitem__(self, index):
        #Create a batch
        #Get a random start pos for each set of frames in the batch
        indices = np.random.randint(self.data_amount,size=self.batch_size)
        
        batch_data = np.zeros((self.batch_size,self.frames_per_sample,)+self.shape)
        batch_labels = np.zeros(self.batch_size)
        
        with h5py.File(self.file_path,'r') as f:
            for i in range(0,self.batch_size):
                #Go up and down from the start position to complete a set of frames for one sample
                label = f["/labels"][indices[i]]
                remaining = self.frames_per_sample
                low = high = indices[i]
                while remaining > 0:
                    if high+1==self.data_amount:
                        low-=1
                        remaining-=1
                    elif f["/labels"][high+1] == label:
                        high+=1
                        remaining-=1
                    else:
                        low-=1
                        remaining-=1
                #add the frames to the current batch data
                batch_data[i]=f["/frames"][low:high]
                batch_labels[i]=label
            return batch_data,batch_labels

    def on_epoch_end(self):
        pass

    
#Reprocess a file for the data generator. 
#This is done in advance so that python doesn't run out of memory when using multiprocessing.
def process_file(in_file_path, out_file_path,frames_per_sample=32, data_amount=300000):
    with h5py.File(in_file_path,'r') as f:
            #Load
            labels = (f['/labels'][:data_amount])
            data = f['/frames/raw'][:data_amount]
            #Change dimensions for networks
            data = (data[(labels>=0)])[...,None]
            labels = labels[(labels>=0)]
            data_amount=len(labels)
            if combine_frames:
                #Array to copy blocks of data into
                temp_data = np.zeros(data.shape)
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
                    if not i == len(data):
                        cur_label=labels[i]
                    #Add block of continous data to new dataset
                    if end-start >= frames_per_sample:
                        temp_data[temp_i:temp_i+end-start]=data[start:end]
                        labels[temp_i:temp_i+end-start]=labels[start:end]
                        temp_i+=end-start
                #trim to new size
                temp_data=temp_data[:temp_i]
                labels=labels[:temp_i]
                data=temp_data
            with h5py.File(out_file_path,'w') as f:
                f.create_dataset("/frames",data=data)
                f.create_dataset("/labels",data=labels)
                del data
                del labels