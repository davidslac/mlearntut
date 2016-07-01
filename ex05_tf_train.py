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
import ex04_tf_train as ex04
import ex02_keras_train as ex02

def get_acc_cmat_for_msg(sess, predict_op, feed_dict, Y, fmtLen):
    predict = sess.run(predict_op, feed_dict=feed_dict)
    confusion_matrix = ex02.get_confusion_matrix_one_hot(predict, Y)
    return ex02.get_acc_cmat_for_msg_from_cmat(confusion_matrix, fmtLen)

def train(train_files, validation_files, saved_model):
    t0 = time.time()
    numOutputs = 2
    training_X, training_Y = ex04.readData(train_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    validation_X, validation_Y = ex04.readData(validation_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    sys.stdout.flush()

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
    model = ex04.build_model(img_placeholder, numOutputs=2)    
    predict_op = tf.nn.softmax(model.final_logits)
    
     ## loss 
    cross_entropy_loss_all = tf.nn.softmax_cross_entropy_with_logits(model.final_logits,
                                                                     labels_placeholder)
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss_all)

    loss = cross_entropy_loss
    
    ## training
    global_step = tf.Variable(0, trainable=False)
    lr = 0.002
    learning_rate = tf.train.exponential_decay(learning_rate=lr,
                                               global_step=global_step,
                                               decay_steps=100,
                                               decay_rate=0.96,
                                               staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)

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

    print(" epoch batch  step tr.sec  loss vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    sys.stdout.flush()
    for epoch in range(3):
        ex04.shuffle_data(training_X, training_Y)
        next_sample_idx = -minibatch_size
        for batch in range(batches_per_epoch):
            step += 1
            next_sample_idx += minibatch_size
            X=training_X[next_sample_idx:(next_sample_idx+minibatch_size),:]
            Y=training_Y[next_sample_idx:(next_sample_idx+minibatch_size),:]
            train_feed_dict = {img_placeholder:X,
                               labels_placeholder:Y}
            t0 = time.time()
            train_loss, jnk = sess.run([loss, train_op], feed_dict=train_feed_dict)
            train_time = time.time()-t0

            msg = " %5d %5d %5d %6.1f %6.3f" % \
                  (epoch, batch, step, train_time, train_loss)

            if step % steps_between_validations == 0:
                t0 = time.time()
                train_acc, cmat_train_rows = get_acc_cmat_for_msg(sess, predict_op, train_feed_dict, Y, fmtLen)
                valid_acc, cmat_valid_rows = get_acc_cmat_for_msg(sess, predict_op, validation_feed_dict, validation_Y, fmtLen)
                valid_time = time.time()-t0
                savemsg = ''
                print('-'*80)
                print('%s %6.1f %5.1f%% %5.1f%% %6.1f | %s | %s | %s' %
                      (msg, valid_time, train_acc*100.0, valid_acc*100.0, 
                       valid_time, cmat_train_rows[0], cmat_valid_rows[0], savemsg))
                for row in range(1,len(cmat_train_rows)):
                    print('%s | %s | %s |' %(' '*(5+6+6+7+7+7+6+6+10),
                                             cmat_train_rows[row],
                                             cmat_valid_rows[row]))
            else:
                print(msg)
            sys.stdout.flush()

                
def predict(predict_files, saved_model):
    pass

def with_graph(train_files, validation_files, predict_files, saved_model, cmd):
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
    sys.stdout.flush()
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

