import tensorflow as tf
from BatchNormalization import BatchNormalization

class SequentialModel(object):
    def __init__(self, img_placeholder, train_placeholder, numOutputs):
        self.img_placeholder = img_placeholder
        self.train_placeholder = train_placeholder
        self.numOutputs = numOutputs
        self.layers = []
        self.names = []
        self.batch_norms = []
        self.vars_to_regularize = []
        self.final_logits = None
        
    def add(self, op, var_to_reg=None):
        self.layers.append(op)
        self.names.append(op.name)
        self.batch_norms.append(None)
        if var_to_reg:
            self.vars_to_regularize.append(var_to_reg)
        return op

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

    def guided_back_prop(self, sess, image, label):
        assert hasattr(self, 'final_logits'), "don't know tensor to back prop from"
        assert image.shape[0]==1, "presently only do guided backprop on batch of size 1"
        assert label >= 0 and label < self.numOutputs, "label=%r must be in [0,%d)" % (label, self.numOutputs)
        assert self.names[0].find('img_float')>=0, 'first layer op is not type cast to float'
        img_tensor = self.layers[0]
        
        relus = [op for op,nm in zip(self.layers,self.names) if nm.find('Relu')>=0]
        assert len(relus)>0, "There are no relu's in the layers."
        
        # start with activation that led to picking the final score
        tensor_above = self.final_logits[0,label]
        tensor_below = relus.pop()

        # take derivative w.r.t relu that preceded it
        deriv_tensor_above_wrt_below = tf.gradients(tensor_above, tensor_below)
        deriv_arr_above_wrt_below = sess.run(deriv_tensor_above_wrt_below,
                                             feed_dict={self.img_placeholder:image,
                                                        self.train_placeholder:False})[0]
        # guided part - keep only positive values in derivative
        deriv_arr_above_wrt_below[deriv_arr_above_wrt_below<0]=0

        # loop through remaining relus, do the same except also evaluate new deriviates
        # at the guided values, with zero-ed out activations
        while len(relus):
            tensor_above = tensor_below
            tensor_below = relus.pop()
            deriv_tensor_above_wrt_below = tf.gradients(tensor_above, tensor_below,
                                                        deriv_arr_above_wrt_below)
            deriv_arr_above_wrt_below = sess.run(deriv_tensor_above_wrt_below,
                                                 feed_dict={self.img_placeholder:image,
                                                            self.train_placeholder:False})[0]
            deriv_arr_above_wrt_below[deriv_arr_above_wrt_below<0]=0

        # one last final deriviate - the input -
        first_relu_after_image = tensor_below
        guided_deriv_of_score_logit_wrt_image = tf.gradients(first_relu_after_image,
                                                             img_tensor,
                                                             deriv_arr_above_wrt_below)
        guided_backprop_arr = sess.run(guided_deriv_of_score_logit_wrt_image,
                                       feed_dict={self.img_placeholder:image,
                                                  self.train_placeholder:False})[0]
        return guided_backprop_arr[0]
        
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
    pool = model.add(tf.nn.max_pool(value=relu, ksize=(1,4,4,1), 
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
