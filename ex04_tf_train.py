from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import random
import math
import numpy as np
import h5py

DATADIR = '/reg/d/ana01/temp/davidsch/ImgMLearnSmall'

import tensorflow as tf

def convert_to_one_hot(labels, numLabels):
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "all values in labels must be in [0,%d)" % numLabels
    return labelsOneHot

def readData(files,
             Xdataset='xtcavimg',
             Ydataset='lasing',
             add_channel='tf',
             Y_onehot_numoutputs=None,
             datadir=DATADIR):
    X = []
    Y = []
    
    for fname in files:
        full_fname = os.path.join(datadir, fname)
        assert os.path.exists(full_fname), "path %s doesn't exist" % full_fname
        h5 = h5py.File(full_fname,'r')
        X.append(h5[Xdataset][:])
        if Ydataset:
            Y.append(h5[Ydataset][:])
            
    X_all = np.concatenate(X)
    nsamples, nrows, ncols = X_all.shape
    nchannels = 1
    if add_channel == 'theano':
        X_all.resize((nsamples, nchannels, nrows, ncols))
    elif add_channel == 'tf':
        X_all.resize((nsamples,nrows, ncols,nchannels))
    elif add_channel not in ['',None]:
        raise Exception("add_channel must be 'tf' or 'theano' or None")

    if not Ydataset:
        return X_all
    
    Y_all = np.concatenate(Y)
    if Y_onehot_numoutputs:
        Y_all = convert_to_one_hot(Y_all, Y_onehot_numoutputs)
    return X_all, Y_all

class SequentialModel(object):
    def __init__(self, img_placeholder, train_placeholder, numOutputs):
        self.img_placeholder = img_placeholder
        self.train_placeholder = train_placeholder
        self.numOutputs = numOutputs
        self.layers = []
        self.names = []
        self.vars_to_regularize = []
        self.final = None
        
    def add(self, op, var_to_reg=None):
        self.layers.append(op)
        self.names.append(op.name)
        if var_to_reg:
            self.vars_to_regularize.append(var_to_reg)
        return op

def build_model(img_placeholder, train_placeholder, numOutputs):

    model = SequentialModel(img_placeholder, train_placeholder, numOutputs)

    img_float = model.add(op=tf.to_float(img_placeholder, name='img_float'))

    ## layer 1
    kernel = tf.Variable(tf.truncated_normal([4,4,1,2], mean=0.0, stddev=0.03))
    conv = model.add(op=tf.nn.conv2d(img_float, kernel, strides=(1,1,1,1), padding='SAME',
                                     data_format='NHWC'), var_to_reg=kernel)
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[2]))
    badd = model.add(op=tf.nn.bias_add(conv, bias))   
    relu = model.add(op=tf.nn.relu(badd))    
    pool = model.add(tf.nn.max_pool(value=relu, ksize=(1,4,4,1), 
                            strides=(1,4,4,1), padding="SAME"))


    ## layer 2
    kernel = tf.Variable(tf.truncated_normal([4,4,2,6],
                                             mean=0.0, stddev=0.03))

    conv = model.add(op=tf.nn.conv2d(pool, kernel, strides=(1,1,1,1), padding='SAME',
                                     data_format='NHWC'), var_to_reg=kernel)    
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[6]))
    badd = model.add(op=tf.nn.bias_add(conv, bias))   
    relu = model.add(op=tf.nn.relu(badd))    
    pool = model.add(tf.nn.max_pool(value=relu, ksize=(1,4,4,1), 
                            strides=(1,4,4,1), padding="SAME"))

    ## flatten
    num_conv_outputs = 1
    for dim in pool.get_shape()[1:].as_list():
        num_conv_outputs *= dim
    conv_outputs = tf.reshape(pool, [-1, num_conv_outputs])

    # layer 3
    weights = tf.Variable(tf.truncated_normal([num_conv_outputs, 40], mean=0.0, stddev=0.03))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[40]))
    xw_plus_b = model.add(tf.nn.xw_plus_b(conv_outputs, weights, bias), var_to_reg=weights)
    nonlinear = tf.nn.relu(xw_plus_b)

    # layer 4
    weights = tf.Variable(tf.truncated_normal([40, 10], mean=0.0, stddev=0.03))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[10]))
    xw_plus_b = model.add(tf.nn.xw_plus_b(nonlinear, weights, bias), var_to_reg=weights)
    nonlinear = tf.nn.relu(xw_plus_b)

    # final layer, logits
    weights = tf.Variable(tf.truncated_normal([10, numOutputs], mean=0.0, stddev=0.03))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[numOutputs]))
    xw_plus_b = model.add(tf.nn.xw_plus_b(nonlinear, weights, bias), var_to_reg=None)

    model.final_logits = xw_plus_b
    return model

              
def shuffle_data(X,Y):
    npseed = int((1<<31)*random.random())
    np.random.seed(npseed)
    np.random.shuffle(X)
    np.random.seed(npseed)
    np.random.shuffle(Y)


def train(train_files, validation_files, saved_model):
    t0 = time.time()
    numOutputs = 2
    training_X, training_Y = readData(train_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    validation_X, validation_Y = readData(validation_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    
    VALIDATION_SIZE = 80
    validation_X = validation_X[0:VALIDATION_SIZE]
    validation_Y = validation_Y[0:VALIDATION_SIZE]

    # EXPLAIN: placeholders
    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
    labels_placeholder = tf.placeholder(tf.float32, 
                                        shape=(None, numOutputs),
                                        name='labels')
    model = build_model(img_placeholder, labels_placeholder, numOutputs=2)    

     ## loss 
    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(model.final_logits,
                                                                     labels_placeholder)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_all)

    ## training
    global_step = tf.Variable(0, trainable=False)
    lr = 0.002
    learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.96,
                                               staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    # EXPLAIN: tensor flow session, also a with way of doing it
    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()
    sess.run(init)
    
    validation_feed_dict = {img_placeholder:validation_X,
                            labels_placeholder:validation_Y}
    

    step = -1
    steps_between_validations = 10

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(max(minibatch_size, VALIDATION_SIZE),10)))

    best_acc = 0.0
    print(" epoch batch  step tr.sec  loss vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    for epoch in range(3):
        shuffle_data(training_X, training_Y)
        next_sample_idx = -minibatch_size
        for batch in range(batches_per_epoch):
            step += 1
            next_sample_idx += minibatch_size
            X=training_X[next_sample_idx:(next_sample_idx+minibatch_size),:]
            Y=training_Y[next_sample_idx:(next_sample_idx+minibatch_size),:]
            train_feed_dict = {img_placeholder:X,
                               labels_placeholder:Y}
            t0 = time.time()
            sess.run(train_op, feed_dict=train_feed_dict)
            train_time = time.time()-t0

            msg = " %5d %5d %5d %6.1f" % \
                  (epoch, batch, step, train_time)
            print(msg)
                
def predict(predict_files, save_fname):
    pass

def with_graph(train_files, validation_files, predict_files, save_fname, cmd):
    if cmd == 'train':
        train(train_files, validation_files, saved_model)
    elif cmd == 'predict':
        predict(predict_files, saved_model)
    else:
        raise Exception(HELP)

if __name__ == '__main__':
    HELP = '''usage: %s cmd, where cmd is one or 'predict' or 'train'.''' % os.path.basename(__file__)
    assert len(sys.argv)==2, "no command given: %s" % HELP
    print("-- imports done, starting main --")
    cmd = sys.argv[1].lower().strip()
    saved_model = 'tf_saved_model'

    train_files = [
        # 3 nolasing files
        'amo86815_mlearn-r069-c0011.h5',
#        'amo86815_mlearn-r069-c0012.h5',
#        'amo86815_mlearn-r069-c0013.h5',
#        'amo86815_mlearn-r069-c0016.h5',
#        'amo86815_mlearn-r069-c0018.h5',
        # 3 lasing files
#        'amo86815_mlearn-r070-c0009.h5',
#        'amo86815_mlearn-r070-c0014.h5',
#        'amo86815_mlearn-r070-c0016.h5',
#        'amo86815_mlearn-r070-c0017.h5',
        'amo86815_mlearn-r070-c0019.h5']

    validation_files = [
        # 1 nolasing files
        'amo86815_mlearn-r069-c0031.h5',
        # 1 lasing files
        'amo86815_mlearn-r070-c0049.h5']

    predict_files = ['amo86815_pred-r073-c0121.h5']

    # EXPLAIN: tf computational graph
    with tf.Graph().as_default():
        with_graph(train_files, validation_files, predict_files, saved_model, cmd)

