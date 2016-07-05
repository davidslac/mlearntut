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

import tensorflow as tf
import MLUtil as util
import TFModel

def train(train_files, validation_files, saved_model):
    t0 = time.time()
    numOutputs = 2
    training_X, training_Y = util.readData(train_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    validation_X, validation_Y = util.readData(validation_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    sys.stdout.flush()
    VALIDATION_SIZE = 80
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
    
    model = TFModel.build_model(img_placeholder, train_placeholder, numOutputs=2)    
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

    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)
    
    validation_feed_dict = {img_placeholder:validation_X,
                            labels_placeholder:validation_Y,
                            train_placeholder:False}
    

    step = -1
    steps_between_validations = 10

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(max(minibatch_size, VALIDATION_SIZE),10)))

    train_ops = [loss, train_op] + model.getTrainOps()
    best_acc = 0.0
    print(" epoch batch  step tr.sec  loss vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    sys.stdout.flush()
    for epoch in range(3):
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
            train_loss = ndarr_train_ops[0]
            train_time = time.time()-t0

            msg = " %5d %5d %5d %6.1f %6.3f" % \
                  (epoch, batch, step, train_time, train_loss)

            if step % steps_between_validations == 0:
                t0 = time.time()
                train_acc, cmat_train_rows = util.get_acc_cmat_for_msg(sess, predict_op, train_feed_dict, Y, fmtLen)
                valid_acc, cmat_valid_rows = util.get_acc_cmat_for_msg(sess, predict_op, validation_feed_dict, validation_Y, fmtLen)
                valid_time = time.time()-t0
                savemsg = ''
                if valid_acc > best_acc:
                    save_path = saver.save(sess, saved_model)
                    best_acc = valid_acc
                    savemsg = ' ** saved in %s' % save_path
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
    t0 = time.time()
    numOutputs = 2

    Xall, Yall = util.readData(train_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    read_time = time.time()-t0
    minibatch_size = 64
    print("-- read %d samples for prediction" % len(Xall))
    sys.stdout.flush()

    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
    train_placeholder = tf.placeholder(tf.bool, name='trainflag')
    
    model = TFModel.build_model(img_placeholder, train_placeholder, numOutputs=2)    
    predict_op = tf.nn.softmax(model.final_logits)
    
    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)

    saver.restore(sess, saved_model)
    print("restored model from %s" % saved_model)
    sys.stdout.flush()

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(minibatch_size,10)))

    idx = -minibatch_size
    Ypred = np.zeros(Yall.shape, dtype=np.float32)
    while idx + minibatch_size < len(Xall):
        idx += minibatch_size
        X=Xall[idx:(idx+minibatch_size)]
        Y=Yall[idx:(idx+minibatch_size)]
        feed_dict={img_placeholder:X,
                   train_placeholder:False}
        Ypred[idx:(idx+minibatch_size)] = sess.run(predict_op, feed_dict=feed_dict)

    cmat = util.get_confusion_matrix_one_hot(Ypred, Yall)

    acc, cmat_rows = util.get_acc_cmat_for_msg_from_cmat(cmat, 3)
    print("Ran predictions. Accuracy: %.2f %d samples" % (acc, len(Ypred)))
    for row in cmat_rows:
        print(row)
    sys.stdout.flush()

def guided_backprop(predict_files, saved_model):
    import matplotlib as mpl
    mpl.rcParams['backend'] = 'TkAgg'
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.show()

    t0 = time.time()
    numOutputs = 2

    Xall, Yall = util.readData(predict_files, 'xtcavimg', 'lasing', 'tf', numOutputs)
    read_time = time.time()-t0
    print("-- read %d samples for prediction" % len(Xall))
    sys.stdout.flush()

    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
    train_placeholder = tf.placeholder(tf.bool, name='trainflag')
    
    model = TFModel.build_model(img_placeholder, train_placeholder, numOutputs=2)    
    predict_op = tf.nn.softmax(model.final_logits)
    
    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)

    saver.restore(sess, saved_model)
    print("restored model from %s" % saved_model)
    sys.stdout.flush()

    for idx in range(len(Xall)):
        X=Xall[idx:idx+1]
        Y=Yall[idx:idx+1]
        feed_dict={img_placeholder:X,
                   train_placeholder:False}
        Ypred = sess.run(predict_op, feed_dict=feed_dict)
        class_pred = np.argmax(Ypred)
        backprop_img_predicted_label = model.guided_back_prop(sess, X, class_pred)[:,:,0]
                
        plt.subplot(1,2,1)
        plt.imshow(X[0,:,:,0], interpolation='none')
        truth=0
        if Y[0,1]: truth=1
        plt.title('raw img %d. pred=%.2f %.2f truth=%d' % (idx, Ypred[0,0], Ypred[0,1], truth))

        plt.subplot(1,2,2)
        plt.imshow(backprop_img_predicted_label, interpolation='none')
        plt.title("guided backprop on predicted label")
        plt.pause(.1)
        
def with_graph(train_files, validation_files, predict_files, saved_model, cmd):
    if cmd == 'train':
        train(train_files, validation_files, saved_model)
    elif cmd == 'predict':
        predict(predict_files, saved_model)
    elif cmd == 'gbprop':
        guided_backprop(predict_files, saved_model)
    else:
        raise Exception(HELP)

if __name__ == '__main__':
    HELP = '''usage: %s cmd, where cmd is one of 'predict', 'train' or 'gbprop'.''' % os.path.basename(__file__)
    assert len(sys.argv)==2, "no command given: %s" % HELP
    print("-- imports done, starting main --")
    cmd = sys.argv[1].lower().strip()
    saved_model = 'tf_saved_model'

    train_files = [
        # nolasing files
        'amo86815_mlearn-r069-c0011.h5',
        'amo86815_mlearn-r069-c0012.h5',
        'amo86815_mlearn-r069-c0013.h5',
        'amo86815_mlearn-r069-c0016.h5',
        'amo86815_mlearn-r069-c0018.h5',
        # lasing files
        'amo86815_mlearn-r070-c0009.h5',
        'amo86815_mlearn-r070-c0014.h5',
        'amo86815_mlearn-r070-c0016.h5',
        'amo86815_mlearn-r070-c0017.h5',
        'amo86815_mlearn-r070-c0019.h5']

    validation_files = [
        # nolasing files
        'amo86815_mlearn-r069-c0031.h5',
        # lasing files
        'amo86815_mlearn-r070-c0049.h5']

    predict_files = ['amo86815_pred-r073-c0121.h5']

    with tf.Graph().as_default():
        with_graph(train_files, validation_files, predict_files, saved_model, cmd)

