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

import ex01_keras_train as ex01
import ex02_keras_train as ex02

from keras.optimizers import SGD

def train(train_files, validation_files, save_fname):
    t0 = time.time()
    training_X, training_Y = ex01.readData(train_files)

    validation_X, validation_Y = ex01.readData(validation_files)
    read_time = time.time()-t0
    minibatch_size = 24
    batches_per_epoch = len(training_X)//minibatch_size
    print("-- read %d samples in %.2fsec. batch_size=%d, %d batches per epoch" %
          (len(training_X)+len(validation_X), read_time, minibatch_size, batches_per_epoch))
    sys.stdout.flush()

    ex01.shuffle_data(validation_X, validation_Y)
    # EXPLAIN: there are 500 rows in file we read, make smaller to speed up validations
    VALIDATION_SIZE = 80
    validation_X = validation_X[0:VALIDATION_SIZE]
    validation_Y = validation_Y[0:VALIDATION_SIZE]

    model = ex01.build_model()

    lr = 0.002
    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    step = -1
    steps_between_validations = 10

    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(max(minibatch_size, VALIDATION_SIZE),10)))

    best_acc = 0.0
    print(" epoch batch  step   loss tr.sec vl.sec tr.acc vl.acc vl.sec  tr.cmat vl.cmat")
    sys.stdout.flush()
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
                train_acc, cmat_train_rows = ex02.get_acc_cmat_for_msg(model, X, Y, fmtLen)
                valid_acc, cmat_valid_rows = ex02.get_acc_cmat_for_msg(model, validation_X, validation_Y, fmtLen)
                valid_time = time.time()-t0
                savemsg = ''
                if valid_acc > best_acc:
                    model.save_weights(save_fname, overwrite=True)
                    best_acc = valid_acc
                    savemsg = ' ** saved'
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

def predict(predict_files, save_fname):
    # EXPLAIN: normally no Y for predict files
    Xall, Yall = ex01.readData(predict_files)
    minibatch_size = 64
    print("read %d samples for prediction" % len(Xall))
    sys.stdout.flush()

    model = ex01.build_model()
    lr = 0.002
    sgd = SGD(lr=lr, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    model.load_weights(save_fname)

    step = -1
    # get decimal places needed to format confusion matrix
    fmtLen = int(math.ceil(math.log(minibatch_size,10)))

    idx = -minibatch_size
    Ypred = np.zeros(Yall.shape, dtype=np.float32)
    while idx + minibatch_size < len(Xall):
        idx += minibatch_size
        X=Xall[idx:(idx+minibatch_size)]
        Y=Yall[idx:(idx+minibatch_size)]
        Ypred[idx:(idx+minibatch_size)] = model.predict(X)
    cmat = ex02.get_confusion_matrix_one_hot(Ypred, Yall)

    acc, cmat_rows = ex02.get_acc_cmat_for_msg_from_cmat(cmat, 3)
    print("Ran predictions. Accuracy: %.2f %d samples" % (acc, len(Ypred)))
    for row in cmat_rows:
        print(row)
    sys.stdout.flush()

if __name__ == '__main__':
    HELP = '''usage: %s cmd, where cmd is one or 'predict' or 'train'.''' % os.path.basename(__file__)
    assert len(sys.argv)==2, "no command given: %s" % HELP
    print("-- imports done, starting main --")
    cmd = sys.argv[1].lower().strip()
    saved_model = 'keras_saved_model.h5'
    train_files = [
        # 3 nolasing files
        'amo86815_mlearn-r069-c0011.h5',
        'amo86815_mlearn-r069-c0012.h5',
        'amo86815_mlearn-r069-c0013.h5',
        'amo86815_mlearn-r069-c0016.h5',
        'amo86815_mlearn-r069-c0018.h5',
        # 3 lasing files
        'amo86815_mlearn-r070-c0009.h5',
        'amo86815_mlearn-r070-c0014.h5',
        'amo86815_mlearn-r070-c0016.h5',
        'amo86815_mlearn-r070-c0017.h5',
        'amo86815_mlearn-r070-c0019.h5']
    validation_files = [
        # 1 nolasing files
        'amo86815_mlearn-r069-c0031.h5',
        # 1 lasing files
        'amo86815_mlearn-r070-c0049.h5']
    # EXPLAIN: good test - 'minibatches' are all lasing, and this is
    # a completely different run.
    predict_files = ['amo86815_pred-r073-c0121.h5']
    if cmd == 'train':
        train(train_files, validation_files, saved_model)
    elif cmd == 'predict':
        predict(predict_files, saved_model)
    else:
        raise Exception(HELP)
