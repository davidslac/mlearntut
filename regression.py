from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import time
import random
import math
import numpy as np
import h5py
from collections import namedtuple

import tensorflow as tf
import MLUtil as util
import TFModel

def getTrainData(mode='test'):
    t0 = time.time()
    Data = namedtuple('Data', 'numOutputs, training_X, training_Y, validation_X, validation_Y')
    numOutputs, training_X, training_Y, validation_X, validation_Y = \
        util.readRegressionForLabel3(mode)
    print("Read %d samples in %.2f sec" % (len(training_X)+len(validation_X), time.time()-t0))
    return Data(numOutputs=numOutputs, 
                training_X=training_X, 
                training_Y=training_Y, 
                validation_X=validation_X, 
                validation_Y=validation_Y)
    
def train(saved_model, trainData=None):
    if trainData is None:
        trainData = getTrainData('test')
    numOutputs, training_X, training_Y, validation_X, validation_Y = \
        trainData.numOutputs, trainData.training_X, trainData.training_Y, \
        trainData.validation_X, trainData.validation_Y

    minibatch_size = 64  
    batches_per_epoch = len(training_X)//minibatch_size
    print("batch size=%d gives %d batches per epoch" % (minibatch_size, batches_per_epoch))
    sys.stdout.flush()

    VALIDATION_SIZE = 128
    util.shuffle_data(validation_X, validation_Y)
    validation_X = validation_X[0:VALIDATION_SIZE]
    validation_Y = validation_Y[0:VALIDATION_SIZE]

    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
 
    labels_placeholder = tf.placeholder(tf.float32, 
                                        shape=(None, numOutputs),
                                        name='labels')
    
    train_placeholder = tf.placeholder(tf.bool, name='trainflag')
    
    model = TFModel.build_regression_model(img_placeholder, train_placeholder, numOutputs)    
                                                                    
    train_op = model.createOptimizerAndGetMinimizationTrainingOp(labels_placeholder=labels_placeholder,
                                                                 learning_rate=0.002, 
                                                                 optimizer_momentum=0.9)

    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)
    
    validation_feed_dict = {img_placeholder:validation_X,
                            labels_placeholder:validation_Y,
                            train_placeholder:False}
    

    step = -1
    steps_between_validations = 10

    train_ops = [model.getModelLoss(), model.getOptLoss(), train_op] + model.getTrainOps()
    best_acc = 0.0
    print(" epoch batch  step tr.sec  mloss  oloss vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    sys.stdout.flush()
    for epoch in range(4):
        util.shuffle_data(training_X, training_Y)
        next_sample_idx = -minibatch_size
        for batch in range(batches_per_epoch):
            step += 1
            next_sample_idx += minibatch_size
            X=training_X[next_sample_idx:(next_sample_idx+minibatch_size),:]
            Y=training_Y[next_sample_idx:(next_sample_idx+minibatch_size),:]
            train_feed_dict = {img_placeholder:X,
                               labels_placeholder:Y,
                               train_placeholder:True}
            t0 = time.time()
            ndarr_train_ops = sess.run(train_ops, feed_dict=train_feed_dict)
            model_loss, opt_loss = ndarr_train_ops[0:2]
            train_time = time.time()-t0

            msg = " %5d %5d %5d %6.1f %6.3f %6.3f" % \
                  (epoch, batch, step, train_time, model_loss, opt_loss)
            print(msg)
            sys.stdout.flush()

    sys.stdout.flush()
    save_path = saver.save(sess, saved_model + '_final')
    print(' ** saved final model in %s' % save_path)
            

def with_graph(saved_model, cmd):
    if cmd == 'train':
        train(saved_model)
    elif cmd == 'predict':
        predict(saved_model)
    elif cmd == 'gbprop':
        guided_backprop(saved_model)
    else:
        raise Exception(HELP)

if __name__ == '__main__':
    HELP = '''usage: %s cmd, where cmd is one of 'predict', 'train' or 'gbprop'.''' % os.path.basename(__file__)
    assert len(sys.argv)==2, "no command given: %s" % HELP
    print("-- imports done, starting main --")
    cmd = sys.argv[1].lower().strip()
    saved_model = 'tf_saved_2color_model'

    with tf.Graph().as_default():
        with_graph(saved_model, cmd)

