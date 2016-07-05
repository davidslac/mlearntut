import os
import numpy as np
import h5py

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
    
def get_acc_cmat_for_msg(sess, predict_op, feed_dict, Y, fmtLen):
    predict = sess.run(predict_op, feed_dict=feed_dict)
    confusion_matrix = ex02.get_confusion_matrix_one_hot(predict, Y)
    return ex02.get_acc_cmat_for_msg_from_cmat(confusion_matrix, fmtLen)
