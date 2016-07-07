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
        util.read2ColorLabelData(mode)
    print("Read %d samples in %.2f sec" % (len(training_X)+len(validation_X), time.time()-t0))
    return Data(numOutputs=numOutputs, 
                training_X=training_X, 
                training_Y=training_Y, 
                validation_X=validation_X, 
                validation_Y=validation_Y)
    
def train(saved_model, trainData=None):
    if trainData is None:
        trainData = getTrainData('all')
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
    
    model = TFModel.build_2color_model(img_placeholder, train_placeholder, numOutputs)    
    predict_op = tf.nn.softmax(model.final_logits)
                                                                    
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

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(max(minibatch_size, VALIDATION_SIZE),10)))

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

            if step % steps_between_validations == 0:
                t0 = time.time()
                train_acc, cmat_train_rows = util.get_acc_cmat_for_msg(sess, predict_op, train_feed_dict, Y, fmtLen)
                valid_acc, cmat_valid_rows = util.get_acc_cmat_for_msg(sess, predict_op, validation_feed_dict, validation_Y, fmtLen)
                print(valid_acc)
                valid_time = time.time()-t0
                savemsg = ''
                if valid_acc > best_acc:
                    save_path = saver.save(sess, saved_model + '_best')
                    best_acc = valid_acc
                    savemsg = ' ** saved best in %s' % save_path
                print('-'*80)
                print('%s %6.1f %5.1f%% %5.1f%% %6.1f | %s | %s | %s' %
                      (msg, valid_time, train_acc*100.0, valid_acc*100.0, 
                       valid_time, cmat_train_rows[0], cmat_valid_rows[0], savemsg))
                for row in range(1,len(cmat_train_rows)):
                    print('%s | %s | %s |' %(' '*(5+6+6+7+7+7+7+6+6+10),
                                             cmat_train_rows[row],
                                             cmat_valid_rows[row]))
            else:
                print(msg)
            sys.stdout.flush()

    sys.stdout.flush()
    save_path = saver.save(sess, saved_model + '_final')
    print(' ** saved final model in %s' % save_path)
            
def predict(saved_model):
    numOutputs, Xall, Yall = util.read2ColorPredictData()
    minibatch_size = 24
    print("-- read %d samples for prediction" % len(Xall))
    sys.stdout.flush()

    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
    train_placeholder = tf.placeholder(tf.bool, name='trainflag')

    model = TFModel.build_2color_model(img_placeholder, train_placeholder, numOutputs)    
    predict_op = tf.nn.softmax(model.final_logits)
    
    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)
    best_saved_model = saved_model + '_best'
    saver.restore(sess, best_saved_model)
    print("restored model from %s" % best_saved_model)
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
        print('predicted on batch %d/%d' % idx/minibatch_size, len(Xall)//minibatch_size)
    cmat = util.get_confusion_matrix_one_hot(Ypred, Yall)

    acc, cmat_rows = util.get_acc_cmat_for_msg_from_cmat(cmat, 3)
    print("Ran predictions. Accuracy: %.2f %d samples" % (acc, len(Ypred)))
    for row in cmat_rows:
        print(row)
    sys.stdout.flush()

def guided_backprop(saved_model):
    import matplotlib as mpl
    mpl.rcParams['backend'] = 'TkAgg'
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.show()

    numOutputs, Xall, Yall = util.read2ColorPredictData()
    print("-- read %d samples for guided backprop" % len(Xall))
    sys.stdout.flush()
    best_saved_model = saved_model + '_best'
    img_placeholder = tf.placeholder(tf.int16,
                                     shape=(None,363,284,1),
                                     name='img')
    train_placeholder = tf.placeholder(tf.bool, name='trainflag')
    
    model = TFModel.build_2color_model(img_placeholder, train_placeholder, numOutputs)    
    predict_op = tf.nn.softmax(model.final_logits)
    
    sess = tf.Session()#config=tf.ConfigProto(intra_op_parallelism_threads = 12))

    init = tf.initialize_all_variables()

    saver = tf.train.Saver()
    
    sess.run(init)

    saver.restore(sess, best_saved_model)
    print("restored model from %s" % best_saved_model)
    sys.stdout.flush()

    guided = True  # set to False to see deriv w.r.t image
    no_colorbar_yet = True
    for idx in range(len(Xall)):
        X=Xall[idx:idx+1]
        Y=Yall[idx:idx+1]
        feed_dict={img_placeholder:X,
                   train_placeholder:False}
        Ypred = sess.run(predict_op, feed_dict=feed_dict)
        class_pred = np.argmax(Ypred)
        if class_pred != 3:
            continue
        backprop_img_predicted_label = model.guided_back_prop(sess, X, class_pred, guided)[:,:,0]
                
        plt.subplot(1,2,1)
        plt.imshow(X[0,:,:,0], interpolation='none', origin = 'lower')
        truth = np.argmax(Y[0,:])
        Ypred_str = map(lambda x: '%.2f'%x, Ypred[0,:])
        plt.title('raw img %d. pred=%s truth=%d' % (idx, Ypred_str, truth))

        plt.subplot(1,2,2)
        plt.imshow(backprop_img_predicted_label, interpolation='none', origin='lower')
        if no_colorbar_yet:
            plt.colorbar()
            no_colorbar_yet = False
        plt.title("guided backprop on predicted label")
        plt.pause(.1)
        if 'q' == raw_input("hit enter for next plot, or q to quit").lower():
            break

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

