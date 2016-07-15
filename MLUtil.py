from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import h5py
import random

def convert_to_one_hot(labels, numLabels):
    labelsOneHot = np.zeros((len(labels), numLabels), dtype=np.int32)
    for label in range(numLabels):
        rowsToSet = np.where(labels==label)[0]
        labelsOneHot[rowsToSet,label] = 1
    assert np.sum(labelsOneHot) == len(labels), "labels must have entries not in [0,%d)" % numLabels
    return labelsOneHot


DATADIR = os.environ.get('DATADIR','/reg/d/ana01/temp/davidsch/ImgMLearnSmall')

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

def read2ColorPredictData():
    validation_files = []
    for run in [70,71]:
        runfiles = glob.glob(os.path.join(DATADIR, 'amo86815_mlearn-r0%d*.h5' % run))
        assert len(runfiles)>0, "no run files found for run=%d, is DATADIR=%s visible?" % (run, DATADIR)
        runfiles.sort()
        validation_files.append(runfiles.pop(0))

    numOutputs = 4

    Xvalid, Yvalid = read2ColorTrainLabelDataFromFiles(validation_files, 
                                                       Xdataset='xtcavimg', 
                                                       Ydataset='acq.enPeaksLabel',
                                                       filter_Y_negone=True,
                                                       add_channel='tf',
                                                       to_one_hot=numOutputs)
    
    return numOutputs, Xvalid, Yvalid
    
def readRegressionForLabel3(mode):
    assert mode in ['all', 'test'], "make mode 'test' or 'all', test is small read"
    validation_files = []
    train_files = []
    for run in [70,71]:
        runfiles = glob.glob(os.path.join(DATADIR, 'amo86815_mlearn-r0%d*.h5' % run))
        assert len(runfiles)>0, "no run files found for run=%d, is DATADIR=%s visible?" % (run, DATADIR)
        runfiles.sort()
        validation_files.append(runfiles.pop(0))
        if mode == 'test':
            train_files.extend(runfiles[0:1])
        else:
            train_files.extend(runfiles[0:])

    numOutputs = 2

    Xtrain, Ytrain = readRegressionFromFiles(train_files)
    Xvalid, Yvalid = readRegressionFromFiles(validation_files) 
    
    return numOutputs, Xtrain, Ytrain, Xvalid, Yvalid

def readRegressionFromFiles(files):
    X = []
    Y = []
    for fname in files:
        h5 = h5py.File(fname,'r')
        currX = h5['xtcavimg'][:]
        e1pos = h5['acq.e1.pos'][:]
        e2pos = h5['acq.e2.pos'][:]
        peaksLabel = h5['acq.enPeaksLabel'][:]
        where3 = peaksLabel == 3
        e1pos = e1pos[where3]
        e2pos = e2pos[where3]
        X.append(currX[where3])
        Y.append(np.transpose(np.vstack((e1pos, e2pos))))
            
    X_all = np.concatenate(X)
    nsamples, nrows, ncols = X_all.shape
    nchannels = 1
    X_all.resize((nsamples,nrows, ncols, nchannels))

    Y_all = np.concatenate(Y)
    return X_all, Y_all

def shuffle_data(X,Y):
    npseed = int((1<<31)*random.random())
    np.random.seed(npseed)
    np.random.shuffle(X)
    np.random.seed(npseed)
    np.random.shuffle(Y)

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
    accuracy = np.trace(confusion_matrix)/float(np.sum(confusion_matrix))
    fmtstr = '%' + str(fmtLen) + 'd'
    cmat_rows = []
    for row in range(confusion_matrix.shape[0]):
        cmat_rows.append(' '.join(map(lambda x: fmtstr % x, confusion_matrix[row,:])))
    return accuracy, cmat_rows

def get_acc_cmat_for_msg(sess, predict_op, feed_dict, Y, fmtLen):
    predict = sess.run(predict_op, feed_dict=feed_dict)
    confusion_matrix = get_confusion_matrix_one_hot(predict, Y)
    return get_acc_cmat_for_msg_from_cmat(confusion_matrix, fmtLen)
