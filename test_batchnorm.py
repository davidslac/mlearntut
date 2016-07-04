from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from BatchNormalization import BatchNormalization

def ndarr_diff(A,B):
    return np.sum(np.abs(A-B))

def assertEqual(A,B,msg):
    assert ndarr_diff(A,B) < 1e-3, msg
    
def with_graph():
    np.random.seed(12342)
    NB = 2 # number batches
    X_batches = [np.random.rand(5,4,3,2).astype(np.float32) for k in range(NB)]
    Y_batches = [np.random.rand(5).astype(np.float32) for k in range(NB)]
    Ainit = np.random.rand(4*3*2,1).astype(np.float32)

    mean_batches = [np.mean(X, axis=(0,1,2)) for X in X_batches]
    std_batches = [np.std(X, axis=(0,1,2)) for X in X_batches]
    
    tensor_X_batches = [tf.constant(X) for X in X_batches]

    X_ph = tf.placeholder(tf.float32, (None,4,3,2))
    Y_ph = tf.placeholder(tf.float32, (None,))
    train_ph=tf.placeholder(tf.bool)

    bnMomentum = 0.5
    bn = BatchNormalization(X_ph, eps=1e-6, mode=0, axis=3, momentum=bnMomentum, train_placeholder=train_ph)
    out = bn.getOp()
    flat = tf.reshape(out, [-1,4*3*2])
    
    A = tf.Variable(Ainit)
    logits = tf.matmul(flat,A)    
    loss = tf.nn.l2_loss(tf.sub(logits,Y_ph))

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.5)
    train_op = optimizer.minimize(loss)
    
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    ##############################################
    ## TEST: INITIALIZE TENSOR FROM NUMPY ARRAYS
    tensor2ndarr_batches = sess.run(tensor_X_batches)
    for arr, tf_arr in zip(X_batches, tensor2ndarr_batches):
        assertEqual(arr, tf_arr, "init tensor from numpy array")

    evalList = [bn.beta, bn.gamma, bn.running_mean, bn.running_std, bn.mean, bn.std]
    evalNames = [tensor.name for tensor in evalList]
    lenEval = len(evalList)
    
    trainEvalList = [train_op] + bn.getTrainOps() + evalList
    trainNames = [tensor.name for tensor in trainEvalList]
    lenTrain = len(trainEvalList)
    idxEval = lenTrain - lenEval
    
    #####################################
    ## TEST: BATCHNORM MODE 0 CONV TRAIN 
    ndarr_train = sess.run(trainEvalList, feed_dict={X_ph:X_batches[0], Y_ph:Y_batches[0], train_ph:True})
    ndarr_eval = ndarr_train[idxEval:]
    _report('-- mode 0 train batch 0', ndarr_eval, evalNames)
    assertEqual(mean_batches[0], ndarr_eval[4], "BatchNormalization mode 0 computation of batch mean during training")
    assertEqual(std_batches[0], ndarr_eval[5], "BatchNormalization mode 0 computation of batch std during training")
    running_mean = (1.0-bnMomentum) * mean_batches[0] 
    running_std = bnMomentum * np.ones_like(std_batches[0]) + (1.0-bnMomentum) * std_batches[0] 
    assertEqual(ndarr_eval[2], running_mean, "BatchNormalization computation of running_mean for mode 0 where it should be getting updated during training")
    assertEqual(ndarr_eval[3], running_std, "BatchNormalization computation of running_std for mode 0 where it should be getting updated during training")

    #####################################
    ## TEST: BATCHNORM MODE 0 CONV TRAIN 
    ndarr_train = sess.run(trainEvalList, feed_dict={X_ph:X_batches[1], Y_ph:Y_batches[1], train_ph:True})
    ndarr_eval = ndarr_train[idxEval:]
    _report('-- mode 0 train batch 1', ndarr_eval, evalNames)
    assertEqual(mean_batches[1], ndarr_eval[4], "BatchNormalization mode 0 computation of batch mean during training")
    assertEqual(std_batches[1], ndarr_eval[5], "BatchNormalization mode 0 computation of batch std during training")
    running_mean *= bnMomentum
    running_mean += (1.0-bnMomentum) * mean_batches[1] 
    running_std *= bnMomentum
    running_std += (1.0-bnMomentum) * std_batches[1] 
    assertEqual(ndarr_eval[2], running_mean, "BatchNormalization computation of running_mean for mode 0 where it should be getting updated during training")
    assertEqual(ndarr_eval[3], running_std, "BatchNormalization computation of running_std for mode 0 where it should be getting updated during training")

    #####################################
    ## TEST: BATCHNORM MODE 0 CONV EVAL 
    ndarr_eval = sess.run(evalList, feed_dict={X_ph:X_batches[0], train_ph:False})
    _report('-- mode 0 test batch 0', ndarr_eval, evalNames)
    assertEqual(running_mean, ndarr_eval[4], "BatchNormalization mode 0 test phase, 'batch mean' should be previous running mean")
    assertEqual(running_std, ndarr_eval[5], "BatchNormalization mode 0 test phase, 'batch std' should be previous running std")
    assertEqual(ndarr_eval[2], running_mean, "BatchNormalization mode 0 test phase, running_mean should not have changed")
    assertEqual(ndarr_eval[3], running_std, "BatchNormalization mode 0 test phase, running_std should not have changed")

def report_test(msg, ndarr_test, names):
    _report(msg, ndarr_test, names)
    
def report_train(msg, ndarr_train, names):
    ndarr_train.pop(0)
    names.pop(0)
    _report(msg, ndarr_train, names)

def _report(msg, ndarrlist, names):
    print("---------------")
    print(msg)
    for nm, arr in zip(names, ndarrlist):
        print("  %20s=%s"%(nm, arr))
        
def test_batchnorm():
    with tf.Graph().as_default():
        with_graph()


if __name__ == '__main__':
    test_batchnorm()
    
