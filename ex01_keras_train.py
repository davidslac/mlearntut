from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import numpy as np
import h5py

# keras defaults to theano, we'll use tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

DATADIR = '/reg/d/ana01/temp/davidsch/ImgMLearnSmall'

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD

def convert_to_one_hot(labels, numLabels):
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "labels must have entries not in [0,%d)" % numLabels
    return labelsOneHot

def readData(files, datadir=DATADIR):
    xtcav = []
    labels = []
    
    for fname in files:
        full_fname = os.path.join(datadir, fname)
        assert os.path.exists(full_fname), "path %s doesn't exist" % full_fname
        h5 = h5py.File(full_fname,'r')
        xtcav.append(h5['xtcavimg'][:])    # EXPLAIN [:]
        labels.append(h5['lasing'][:])
    xtcav_all = np.concatenate(xtcav)
    nsamples, nrows, ncols = xtcav_all.shape
    nchannels = 1
    xtcav_all.resize((nsamples, nchannels, nrows, ncols))  # EXPLAIN add channel
    return xtcav_all, convert_to_one_hot(np.concatenate(labels), 2)  # EXPLAIN: one_hot

def build_model():
    model = Sequential()
    model.add(Convolution2D(2,4,4, border_mode='same', 
                            input_shape=(1,363, 284), 
                            bias=False))                # EXPLAIN bias=False
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1,    # EXPLAIN axis
                                 momentum=0.9, 
                                 beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))

    model.add(Convolution2D(6,4,4, border_mode='same', 
                            bias=False))
    model.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, 
                                 momentum=0.9, 
                                 beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    
    model.add(Flatten())
    
    model.add(Dense(40, bias=False))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=1,          # EXPLAIN mode=1
                                 momentum=0.9,
                                 beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))
    
    model.add(Dense(10, bias=False))
    model.add(BatchNormalization(epsilon=1e-06, 
                                 mode=1, 
                                 momentum=0.9, 
                                 beta_init='zero', gamma_init='one'))
    model.add(Activation('relu'))
    
    model.add(Dense(2, bias=True))
    model.add(Activation('softmax'))

    return model
              
def shuffle_data(X,Y):
    npseed = int((1<<31)*random.random())
    np.random.seed(npseed)
    np.random.shuffle(X)
    np.random.seed(npseed)
    np.random.shuffle(Y)

if __name__ == '__main__':
    print("-- imports done, starting main --")
    t0 = time.time()
    training_X, training_Y = readData([
        # 3 nolasing files
        'amo86815_mlearn-r069-c0011.h5',
        'amo86815_mlearn-r069-c0012.h5',
        'amo86815_mlearn-r069-c0013.h5',
        # 3 lasing files
        'amo86815_mlearn-r070-c0009.h5',
        'amo86815_mlearn-r070-c0014.h5',
        'amo86815_mlearn-r070-c0016.h5'])
    validation_X, validation_Y = readData([
        # 1 nolasing files
        'amo86815_mlearn-r069-c0031.h5',
        # 1 lasing files
        'amo86815_mlearn-r070-c0029.h5'])
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    
    shuffle_data(validation_X, validation_Y)
    model = build_model()
    
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    # EXPLAIN: why not use fit? 
    # fit takes X,Y numpy arrays in memory, easier, but doesn't scale
    for epoch in range(3):
        shuffle_data(training_X, training_Y)
        next_sample_idx = -minibatch_size
        for batch_number in range(batches_per_epoch):
            next_sample_idx += minibatch_size
            X=training_X[next_sample_idx:(next_sample_idx+minibatch_size),:]
            Y=training_Y[next_sample_idx:(next_sample_idx+minibatch_size),:]
            t0 = time.time()
            train_loss = model.train_on_batch(X,Y)
            train_time = time.time()-t0
            print("epoch=%d batch=%d train_loss=%.3f train_step_time=%.2f" % 
                  (epoch, batch_number, train_loss, train_time))
    # EXPLAIN: better to have minibatch size evenly divide number train samples
