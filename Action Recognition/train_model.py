from __future__ import absolute_import, division, print_function, unicode_literals

import click
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

import DataGenerator as DG
from DataGenerator import DataGenerator

from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard

#Progress bar fix: use callbacks=[Logger.JupyterProgbarLogger()] in fit method
#verbose=0 is also required
import JupyterProgbarLogger as Logger

import numpy as np
import random
import math

from datetime import datetime
from shutil import copy

import h5py
import multiprocessing as mp

@click.group()
def cli():
    pass

@cli.command(name = 'train')
@click.argument("infile", type = click.Path())
@click.argument("outdir", type = click.Path())
@click.option("--data-amount", type=int, default=1000000, help="Amount of training data")
@click.option("--data-offset", type=int, default=0, help="Training data offset")
@click.option("--validation-amount", type=int, default=100000, help="Amount of validation data")
@click.option("--validation-offset", type=int, default=1000000, help="Validation data offset")
@click.option("--depth", type=int, default=10, help="Depth of frames per sample")
@click.option("--batch-size", type=int, default=8, help="Batch Size")
@click.option("--slide", type=int, default=5, help="Sliding window amount")
@click.option("--epochs", type=int, default=30, help="Number of training epochs")
@click.option("--verbosity", type=int, default=1, help="Verbosity")
@click.option("--dropout", type=float, default=0.0, help="Dropout Chance")
@click.option("--filters", type=int, default=8, help="Number of convolutional filters")
@click.option("--optimizer", default='SGD', help="Optimizer to use (RMSProp or SGD)")
@click.option("--lstm-units", type=int, default=512, help="LSTM units")
def train_model(infile,
                outdir,
                data_amount=1000000,
                data_offset=0,
                validation_amount=100000,
                validation_offset=1000000,
                batch_size=8,
                slide = 5,
                epochs=15,
                verbosity = 1,
                tuner=None,
                depth = 10,
                input_shape=(80, 80, 1),
                stride_length=(1, 1),
                kernel=(3,3),
                kernel_initializer='glorot_uniform',
                activation=layers.Activation('relu'),
                output_activation=layers.Activation('softmax'),
                dropout=0,
                padding='same',
                filters=16,
                optimizer = 'SGD',
                lstm_units=512):
    ###FIX NUMPY LOAD FOR DICTIONARIES
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    mp.set_start_method("spawn",force=True)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    ###Tensorflow session
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    outdir = outdir % os.environ["SLURM_JOBID"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    #Initialize generators
    data_gen = DataGenerator(infile,
                             data_amount=data_amount,
                             batch_size=batch_size,
                             frames_per_sample=depth,
                             offset=data_offset,
                             sliding_window=slide)
    validation_gen = DataGenerator(infile,
                                   data_amount=validation_amount,
                                   batch_size=batch_size,
                                   frames_per_sample=depth,
                                   offset=validation_offset,
                                   sliding_window=slide)
    name = "LSTM CNN"
    if depth > 1:
        input_shape = (depth,)+input_shape
    inputs = layers.Input(shape=input_shape)
    x = inputs
    conv_parameters = {
        'padding': padding,
        'strides': stride_length,
        'kernel_initializer': kernel_initializer
    }
    # encode net
    x = layers.TimeDistributed(layers.Conv2D(filters, kernel, **conv_parameters), input_shape=input_shape)(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), padding=padding))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*2, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*2, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), padding=padding))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*4, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*4, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), padding=padding))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*8, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.Conv2D(filters*8, kernel, **conv_parameters))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2), padding=padding))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    if dropout>0:
        x = layers.SpatialDropout1D(dropout)(x)
    x = layers.LSTM(lstm_units,return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = output_activation(x)
    model = keras.models.Model(inputs, output)
    if optimizer == 'RMSProp':
        optimizer = keras.optimizers.RMSprop(lr=1e-4)
    else:
        optimizer=keras.optimizers.SGD(
            learning_rate=1e-4,
            momentum=.9,
            nesterov=True,
            decay=1e-6
        )
    model.compile(optimizer = optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [keras.callbacks.ModelCheckpoint(filepath= (outdir+'model_progress_{epoch:02d}.h5'),save_best_only=True),
                 Logger.JupyterProgbarLogger(count_mode='steps',notebook=False)]
    history = model.fit_generator(generator=data_gen,
                    validation_data=validation_gen,
                    epochs=epochs,
                    verbose=verbosity,
                    use_multiprocessing=True,
                    workers=16,
                    callbacks=callbacks)
    print(model.summary())
    
@cli.command(name = 'continue')
@click.argument("infile", type = click.Path())
@click.argument("outdir", type = click.Path())
@click.argument("initial", type = click.Path())
@click.option("--data-amount", type=int, default=1000000, help="Amount of training data")
@click.option("--data-offset", type=int, default=0, help="Training data offset")
@click.option("--validation-amount", type=int, default=100000, help="Amount of validation data")
@click.option("--validation-offset", type=int, default=1000000, help="Validation data offset")
@click.option("--batch-size", type=int, default=8, help="Batch Size")
@click.option("--slide", type=int, default=5, help="Sliding window amount")
@click.option("--epochs", type=int, default=30, help="Number of training epochs")
@click.option("--verbosity", type=int, default=1, help="Verbosity")
def continue_model(infile,
                outdir,
                initial,
                data_amount=1000000,
                data_offset=0,
                validation_amount=100000,
                validation_offset=1000000,
                batch_size=8,
                slide = 5,
                epochs=15,
                verbosity = 1):
    ###FIX NUMPY LOAD FOR DICTIONARIES
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    mp.set_start_method("spawn",force=True)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    ###Tensorflow session
    #config = tf.compat.v1.ConfigProto()
    #config.gpu_options.allow_growth = True
    #tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    
    outdir = outdir % os.environ["SLURM_JOBID"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model = keras.models.load_model(initial)
    if len(model.layers[0].output_shape[0])==4:
        depth = 1
    else:
        depth = model.layers[0].output_shape[0][1]
    multiple_inputs=False
    if len(model._input_layers) > 1:
        multiple_inputs=True
    #Initialize generators
    data_gen = DataGenerator(infile,
                             data_amount=data_amount,
                             batch_size=batch_size,
                             frames_per_sample=depth,
                             offset=data_offset,
                             sliding_window=slide,
                             labels_structured=multiple_inputs)
    validation_gen = DataGenerator(infile,
                                   data_amount=validation_amount,
                                   batch_size=batch_size,
                                   frames_per_sample=depth,
                                   offset=validation_offset,
                                   sliding_window=slide,
                                   labels_structured=multiple_inputs)
    #os.remove(initial)
    callbacks = [keras.callbacks.ModelCheckpoint(filepath= (outdir+'model_progress_{epoch:02d}.h5')),
                 Logger.JupyterProgbarLogger(count_mode='steps',notebook=False)]
    history = model.fit_generator(generator=data_gen,
                    validation_data=validation_gen,
                    epochs=epochs,
                    verbose=verbosity,
                    use_multiprocessing=True,
                    workers=16,
                    callbacks=callbacks)
    print(model.summary())
    
if __name__ == '__main__':
          cli()
