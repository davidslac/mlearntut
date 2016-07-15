from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from BatchNormalization import BatchNormalization

###############
class SequentialModel(object):
    def __init__(self, img_placeholder, train_placeholder, numOutputs, regFn='L2', regWeight=None):
        self.img_placeholder = img_placeholder
        self.train_placeholder = train_placeholder
        self.numOutputs = numOutputs
        assert regFn in [None, 'L2', 'L1'], "regFn must be None, L2, or L1, but it is %s" % regFn
        self.regFn = regFn
        self.regWeight = regWeight
        self.layers = []
        self.names = []
        self.batch_norms = []
        self.regTerm = None
        self.final_logits = None

    def _regFnToUse(self, regFn):
        if regFn is not None:
            assert regFn in ['L2', 'L1'], 'regFn must be None, L1 or L2'
            return regFn
        return self.regFn
            
    def add(self, op, var_to_reg=None, regFn=None, regWeight=None):
        self.layers.append(op)
        self.names.append(op.name)
        self.batch_norms.append(None)
        ## support a list of variables, or just one variable
        if var_to_reg is not None:
            if not isinstance(var_to_reg, list):
                var_to_reg = [var_to_reg]
            for var in var_to_reg:
                if regWeight is None:
                    regWeight = self.regWeight
                if regWeight is not None:
                    if self._regFnToUse(regFn) == 'L2':
                        term = 0.5 * tf.reduce_sum(tf.mul(var, var))
                    elif self._regFnToUse(regFn) == 'L1':
                        term = tf.reduce_sum(tf.abs(var))
                    term *= regWeight
                    if self.regTerm is None:
                        self.regTerm = term
                    else:
                        self.regTerm += term
        return op

    def getRegTerm(self):
        return self.regTerm

    def add_batch_norm(self, eps, mode, axis=-1, momentum=0.9, beta_init='zero',  gamma_init='one'):
        assert len(self.layers)>0, "no op to apply batch to"
        last_op = self.layers[-1]
        assert beta_init == 'zero', 'only do beta_init==0'
        assert gamma_init == 'one', 'only do gamma_init==1'
        bn = BatchNormalization(inputTensor=last_op, eps=eps, mode=mode,
                                axis=axis, momentum=momentum, train_placeholder=self.train_placeholder)
        op = bn.getOp()
        self.layers.append(op)
        self.names.append('batchnorm')
        self.batch_norms.append(bn)
        return op

    def getTrainOps(self):
        ops = []
        for bn in self.batch_norms:
            if bn is None:
                continue
            ops.extend(bn.getTrainOps())
        return ops

    def createOptimizerAndGetMinimizationTrainingOp(self, 
                                                    labels_placeholder,
                                                    learning_rate,
                                                    optimizer_momentum,
                                                    decay_steps=100,
                                                    decay_rate=0.96,
                                                    staircase=True):

        diff = tf.sub(self.final_logits, labels_placeholder)
        self.model_loss = tf.reduce_mean(diff * diff)

        self.optimizer_loss = self.model_loss
        if self.regTerm is not None:
            self.optimizer_loss += self.regTerm
            
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=staircase)

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, 
                                                    momentum=optimizer_momentum)
        self.train_op = self.optimizer.minimize(self.optimizer_loss, global_step=self.global_step)
        return self.train_op

    def getModelLoss(self):
        return self.model_loss

    def getOptLoss(self):
        return self.optimizer_loss

    def guided_back_prop(self, sess, image, label, do_guided=True):
        assert hasattr(self, 'final_logits'), "don't know tensor to back prop from"
        assert image.shape[0]==1, "presently only do guided backprop on batch of size 1"
        assert label >= 0 and label < self.numOutputs, \
            "label=%r must be in [0,%d)" % (label, self.numOutputs)
        assert self.names[0].find('img_float')>=0, 'first layer op is not type cast to float'

        feed_dict = {self.img_placeholder:image,
                     self.train_placeholder:False}

        # start with activation that led to picking the final score
        ys = self.final_logits[0,label]
        ys_name = 'final'
        grad_ys = np.ones([1], dtype=np.float32)

        # There is some currently bug with my batch normalization, for some
        # reason derivatives of batch outputs with respect to the layer below are
        # turning into nan's, but only if it is batch over a dense layer, it is
        # fine for the covnets, something about the shape maybe? 

        # to workaround, if we see nan's for such a layer, we throw away the
        # nan gradients and just use the previous gradient
        for xIdx in range(len(self.layers)-1,-1,-1):
            xs = self.layers[xIdx]
            xs_name = self.names[xIdx]
            grad_ys_wrt_xs = tf.gradients(ys, xs, grad_ys)            
            new_grad_ys = sess.run(grad_ys_wrt_xs, feed_dict=feed_dict)[0]
            if np.any(np.isnan(new_grad_ys)):
                assert new_grad_ys.shape==grad_ys.shape, \
                    ("nan's showing up in gradient for layer xIdx=%d,"+ \
                    "ys_name=%s xs_name=%s, new_shape=%s != old_shape=%s") % \
                    (xIdx, ys_name, xs_name, new_grad_ys.shape, grad_ys.shape)
            else:
                grad_ys = new_grad_ys
            # guided part:
            if do_guided and ys_name.lower().find('relu')>=0:
                # 'Striving for Simplicity' paper 
                # http://arxiv.org/pdf/1412.6806v3.pdf
                # says that for guided backprop,
                # for relu, mask out values for either of these cases:
                # * negative values in the top gradient 
                #   (derivate with w.r.t. x when y=relu?)
                # * negative values of the bottom 
                #   (when the x, going into the relu is < 0?) 
                grad_ys[grad_ys<0]=0.0
                # should we do this also?
                xarr = sess.run(xs, feed_dict=feed_dict)
                grad_ys[xarr<0]=0.0
            ys = xs
            ys_name = xs_name
        return grad_ys[0]

    def _guided_back_prop(self, sess, image, label):
        # this is the guided back prop we would like to use, if not for the
        # batchnorm bug
        assert hasattr(self, 'final_logits'), "don't know tensor to back prop from"
        assert image.shape[0]==1, "presently only do guided backprop on batch of size 1"
        assert label >= 0 and label < self.numOutputs, \
            "label=%r must be in [0,%d)" % (label, self.numOutputs)
        assert self.names[0].find('img_float')>=0, 'first layer op is not type cast to float'

        feed_dict = {self.img_placeholder:image,
                     self.train_placeholder:False}

        # start with activation that led to picking the final score
        ys = self.final_logits[0,label]
        grad_ys = np.ones([1], dtype=np.float32)
        relus = [op for op,nm in zip(self.layers, self.names) \
                 if nm.lower().find('relu')>=0]
        assert self.names[0].find('img_float')>=0, 'expected typecast to img float for first layer, it is %s' % self.names[0]
        op_list = [self.layers[0]] + relus
        while len(op_list):
            xs = op_list.pop()
            grad_ys_wrt_xs = tf.gradients(ys, xs, grad_ys)
            grad_ys = sess.run(grad_ys_wrt_xs, feed_dict=feed_dict)[0]
            grad_ys[grad_ys<0]=0.0
            # but do we need to also 0 out based on what is below?
            ys = xs
        return grad_ys[0]
        
def build_model(img_placeholder, train_placeholder, numOutputs):

    model = SequentialModel(img_placeholder, train_placeholder, numOutputs)

    img_float = model.add(op=tf.to_float(img_placeholder, name='img_float'))

    ## layer 1
    kernel = tf.Variable(tf.truncated_normal([4,4,1,2], mean=0.0, stddev=0.03))
    conv = model.add(op=tf.nn.conv2d(img_float, kernel, strides=(1,1,1,1), padding='SAME',
                                     data_format='NHWC'), var_to_reg=kernel)
    batch = model.add_batch_norm(eps=1e-06, mode=0, axis=3, momentum=0.9, beta_init='zero',  gamma_init='one')
    relu = model.add(op=tf.nn.relu(batch))    
    pool = model.add(tf.nn.max_pool(value=relu, ksize=(1,4,4,1), 
                            strides=(1,4,4,1), padding="SAME"))


    ## layer 2
    kernel = tf.Variable(tf.truncated_normal([4,4,2,6],
                                             mean=0.0, stddev=0.03))

    conv = model.add(op=tf.nn.conv2d(pool, kernel, strides=(1,1,1,1), padding='SAME',
                                     data_format='NHWC'), var_to_reg=kernel)    
    batch = model.add_batch_norm(eps=1e-06, mode=0, axis=3, momentum=0.9, beta_init='zero',  gamma_init='one')
    relu = model.add(op=tf.nn.relu(batch))    
    pool = model.add(op=tf.nn.max_pool(value=relu, ksize=(1,4,4,1), 
                                       strides=(1,4,4,1), padding="SAME"))

    ## flatten
    num_conv_outputs = 1
    for dim in pool.get_shape()[1:].as_list():
        num_conv_outputs *= dim
    conv_outputs = tf.reshape(pool, [-1, num_conv_outputs])

    # layer 3
    weights = tf.Variable(tf.truncated_normal([num_conv_outputs, 40], mean=0.0, stddev=0.03))
    xw = model.add(tf.matmul(conv_outputs, weights), var_to_reg=weights)
    batch = model.add_batch_norm(eps=1e-06, mode=1, momentum=0.9, beta_init='zero',  gamma_init='one')
    nonlinear = model.add(op=tf.nn.relu(batch))

    # layer 4
    weights = tf.Variable(tf.truncated_normal([40, 10], mean=0.0, stddev=0.03))
    xw = model.add(tf.matmul(nonlinear, weights), var_to_reg=weights)
    batch = model.add_batch_norm(eps=1e-06, mode=1, momentum=0.9, beta_init='zero',  gamma_init='one')
    nonlinear = model.add(op=tf.nn.relu(batch))

    # final layer, logits
    weights = tf.Variable(tf.truncated_normal([10, numOutputs], mean=0.0, stddev=0.03))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[numOutputs]))
    xw_plus_b = model.add(tf.nn.xw_plus_b(nonlinear, weights, bias), var_to_reg=None)

    model.final_logits = xw_plus_b
    return model

def build_regression_model(img_placeholder, train_placeholder, numOutputs):
    regWeight = 0.01
    model = SequentialModel(img_placeholder, train_placeholder, numOutputs, regWeight=regWeight)
    img_float = model.add(op=tf.to_float(img_placeholder, name='img_float'))

    ## layer 1
    ch01 = 16
    xx = 8
    yy = 8
    kernel = tf.Variable(tf.truncated_normal([xx,yy,1,ch01], mean=0.0, stddev=0.03))
    conv = model.add(op=tf.nn.conv2d(img_float, kernel, strides=(1,1,1,1), 
                                     padding='SAME',data_format='NHWC'),
                     var_to_reg=kernel)
    batch = model.add_batch_norm(eps=1e-06, mode=0, axis=3, momentum=0.9)
    relu = model.add(op=tf.nn.relu(batch))
    pool = model.add(op=tf.nn.max_pool(value=relu, ksize=(1,2,2,1), 
                                       strides=(1,3,3,1), padding='SAME'))

    ## layer 2
    ch02=16
    xx=6
    yy=6
    kernel = tf.Variable(tf.truncated_normal([xx,yy,ch01,ch02], mean=0.0, stddev=0.03))
    conv = model.add(op=tf.nn.conv2d(pool, kernel, strides=(1,1,1,1), 
                                     padding='SAME',data_format='NHWC'),
                     var_to_reg=kernel)
    batch = model.add_batch_norm(eps=1e-06, mode=0, axis=3, momentum=0.9)
    relu = model.add(op=tf.nn.relu(batch))
    pool = model.add(op=tf.nn.max_pool(value=relu, ksize=(1,3,3,1), 
                                       strides=(1,3,3,1), padding='SAME'))
    
    ## layer 3
    ch03 = 16
    xx = 6
    yy = 6
    kernel = tf.Variable(tf.truncated_normal([xx,yy,ch02,ch03], mean=0.0, stddev=0.03))
    conv = model.add(op=tf.nn.conv2d(pool, kernel, strides=(1,1,1,1), 
                                     padding='SAME',data_format='NHWC'),
                     var_to_reg=kernel)
    batch = model.add_batch_norm(eps=1e-06, mode=0, axis=3, momentum=0.9)
    relu = model.add(op=tf.nn.relu(batch))
    pool = model.add(op=tf.nn.max_pool(value=relu, ksize=(1,3,3,1), 
                                       strides=(1,3,3,1), padding='SAME'))
    
    ## flatten
    num_conv_outputs = 1
    for dim in pool.get_shape()[1:].as_list():
        num_conv_outputs *= dim
    model.conv_outputs = tf.reshape(pool, [-1, num_conv_outputs])
    
    ## layer 4
    hidden04 = 48
    weights = tf.Variable(tf.truncated_normal([num_conv_outputs, hidden04], mean=0.0, stddev=0.03))
    xw = model.add(tf.matmul(model.conv_outputs, weights), var_to_reg=weights)
    batch = model.add_batch_norm(eps=1e-06, mode=1, momentum=0.9)
    relu = model.add(op=tf.nn.relu(batch))

    ## layer 5
    hidden05 = 32
    weights = tf.Variable(tf.truncated_normal([hidden04, hidden05], mean=0.0, stddev=0.03))
    xw = model.add(tf.matmul(relu, weights), var_to_reg=weights)
    batch = model.add_batch_norm(eps=1e-06, mode=1, momentum=0.9)
    relu = model.add(op=tf.nn.relu(batch))

    # final layer, logits
    weights = tf.Variable(tf.truncated_normal([hidden05, numOutputs], mean=0.0, stddev=0.03))
    bias = tf.Variable(tf.constant(value=0.0, dtype=tf.float32, shape=[numOutputs]))
    xw_plus_b = model.add(tf.nn.xw_plus_b(relu, weights, bias), 
                          var_to_reg = [weights, bias], regWeight=0.2*regWeight)
    
    model.final_logits = xw_plus_b
    return model
                         
