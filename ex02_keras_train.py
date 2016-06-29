from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random
import math
import numpy as np
import h5py

# EXPLAIN: reuse previous exercise
import ex01_keras_train as ex01

from keras.optimizers import SGD

def get_confusion_matrix_one_hot(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix

def get_acc_cmat_for_msg(model, X, Y, fmtLen):
    predict = model.predict(X)
    confusion_matrix = get_confusion_matrix_one_hot(predict, Y)
    return get_acc_cmat_for_msg_from_cmat(confusion_matrix, fmtLen)

def get_acc_cmat_for_msg_from_cmat(confusion_matrix, fmtLen):
    accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
    fmtstr = '%' + str(fmtLen) + 'd'
    cmat_rows = []
    for row in range(confusion_matrix.shape[0]):
        cmat_rows.append(' '.join(map(lambda x: fmtstr % x, confusion_matrix[row,:])))
    return accuracy, cmat_rows

if __name__ == '__main__':
    print("-- imports done, starting main --")
    t0 = time.time()
    training_X, training_Y = ex01.readData([
        # 3 nolasing files
        'amo86815_mlearn-r069-c0011.h5',
        'amo86815_mlearn-r069-c0012.h5',
        'amo86815_mlearn-r069-c0013.h5',
        # 3 lasing files
        'amo86815_mlearn-r070-c0009.h5',
        'amo86815_mlearn-r070-c0014.h5',
        'amo86815_mlearn-r070-c0016.h5'])
    validation_X, validation_Y = ex01.readData([
        # 1 nolasing files
        'amo86815_mlearn-r069-c0031.h5',
        # 1 lasing files
        'amo86815_mlearn-r070-c0029.h5'])
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    
    ex01.shuffle_data(validation_X, validation_Y)
    # EXPLAIN: there are 500 rows in file we read, make smaller to speed up validations
    VALIDATION_SIZE = 30
    validation_X = validation_X[0:VALIDATION_SIZE]
    validation_Y = validation_Y[0:VALIDATION_SIZE]

    model = ex01.build_model()
    
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    step = -1
    steps_between_validations = 3

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(max(minibatch_size, VALIDATION_SIZE),10)))

    print(" epoch batch  step   loss tr.sec vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    for epoch in range(3):
        ex01.shuffle_data(training_X, training_Y)
        next_sample_idx = -minibatch_size
        for batch in range(batches_per_epoch):
            step += 1
            next_sample_idx += minibatch_size
            X=training_X[next_sample_idx:(next_sample_idx+minibatch_size),:]
            Y=training_Y[next_sample_idx:(next_sample_idx+minibatch_size),:]
            t0 = time.time()
            train_loss = model.train_on_batch(X,Y)
            train_time = time.time()-t0
            msg = " %5d %5d %5d %6.3f %6.1f" % \
                        (epoch, batch, step, train_loss, train_time)
            if step % steps_between_validations == 0:
                t0 = time.time()
                train_acc, cmat_train_rows = get_acc_cmat_for_msg(model, X, Y, fmtLen)
                valid_acc, cmat_valid_rows = get_acc_cmat_for_msg(model, validation_X, validation_Y, fmtLen)
                valid_time = time.time()-t0
                print('-'*80)
                print('%s %6.1f %5.1f%% %5.1f%% %6.1f | %s | %s |' %
                      (msg, valid_time, train_acc*100.0, valid_acc*100.0, 
                       valid_time, cmat_train_rows[0], cmat_valid_rows[0]))
                for row in range(1,len(cmat_train_rows)):
                    print('%s | %s | %s |' %(' '*(5+6+6+7+7+7+6+6+10),
                                             cmat_train_rows[row],
                                             cmat_valid_rows[row]))
            else:
                print(msg)
