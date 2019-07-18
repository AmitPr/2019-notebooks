import numpy as np
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    #Initialize
    def __init__(self, file_path, shape=(32,80,80,1), data_amount=300000, batch_size=32, shuffle=True):
        self.combine_frames=(not blocksize==1)
        with h5py.File(file_path,'r') as f:
            #Load
            self.labels = (f['/labels'][:data_amount])
            self.data = f['/frames/raw'][:data_amount]
            #Change dimensions for networks
            self.data = (self.data[(self.labels>=0)])[...,None]
            self.labels = self.labels[(self.labels>=0)]
        if combine_frames:
            #Array to copy blocks of data into
            temp_data = np.zeros((data_amount,)+shape)
            #current index in temp_data
            temp_i = 0
            i = 0
            cur_label = labels[0]
            while i < data_amount:
                start = i
                while labels[i]==cur_label:
                    i+=1
                    if i == len(data):
                        break
                end = i
                if not i == len(data):
                    cur_label=labels[i]
                block_length = end-start
                if block_length >= shape[0]:
                    temp_data[temp_i:temp_i+block_length]=data[start:end]
                    temp_i+=block_length
                    labels[temp_i:temp_i+block_length]=labels[start:end]
            temp_data=temp_data[:temp_i]
            labels=labels[:temp_i]
            self.data=temp_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)