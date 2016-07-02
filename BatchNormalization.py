import tensorflow as tf

class BatchNormalization(object):
    '''
    '''
    def __init__(self, inputTensor, eps, mode, axis, momentum, train_placeholder):
        '''modes - follow keras for 0,1,2 and add 3
        0: featurewise: train - per batch
                        test  - moving average
        1: samplewise   train - per batch
                        test  - per batch
        2: featurewise: train - per batch
                        test  - per batch
        3: samplewise   train - per batch
                        test  - moving average
        '''
        input_shape = inputTensor.get_shape().as_list()
        assert mode in [0,1,2,3], "mode param invalid"
        self.mode = mode
        if mode in [0,2]:
            assert axis > 0, "specify axis>0 (channel dim) for feature maps"
            assert axis == len(input_shape)-1, "presently mode 0/2 only support last dim for axis - makes broadcasting simple"
            depth = input_shape[axis]
            reduction_indices = [idx for idx in range(len(input_shape)) if idx != axis]
        elif mode in [1,3]:
            assert len(input_shape)==2, "sample wise mode=0 is for flattened 2D input tensor, not convnets"
            assert axis in [-1,1], "do not specify axis for mode 1/3, or set it to 1"
            depth = input_shape[1]
            reduction_indices = [0]

        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=True)
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=True)

        self.running_mean = tf.Variable(tf.zeros(depth), trainable=False, name='running_mean') 
        self.running_std = tf.Variable(tf.ones(depth), trainable=False, name='running_std') 

        if mode in [0,3]:
            mean, std = tf.cond(train_placeholder,
                                UseBatchAndUpdateAvg(inputTensor, 
                                                     reduction_indices,
                                                     momentum,
                                                     self.running_mean, self.running_std),
                                UseAvg(self.running_mean, self.running_std))
        elif mode in [1,2]:
            mean, std = calcBatchStats(inputTensor, reduction_indices)

        # tensorflow will broadcast over the first dimensions, why we enforce axis as the last dim
        meanNorm = tf.sub(inputTensor, mean)
        stdNorm = tf.div(meanNorm, tf.add(eps, std))
        self.op = tf.add(tf.mul(self.gamma, stdNorm), self.beta)

    def getOp(self):
        return self.op


def calcBatchStats(inputTensor, reduction_indices):
        input_shape = inputTensor.get_shape().as_list()
        assert reduction_indices == range(len(input_shape)-1), "only support last dim as axis to make broadcast simple"
        batchMean, batchVar = tf.nn.moments(inputTensor, axes=reduction_indices)
        batchStd = tf.sqrt(batchVar)
        return batchMean, batchStd



class UseBatchAndUpdateAvg(object):
    def __init__(self, inputTensor, 
                 reduction_indices, momentum, running_mean, running_std):
        batchMean, batchStd = calcBatchStats(inputTensor, reduction_indices)
        new_running_mean = tf.add(tf.mul(momentum, running_mean), (1.0-momentum)*batchMean)
        new_running_std = tf.add(tf.mul(momentum, running_std), (1.0-momentum)*batchStd)
        assign_running_mean = running_mean.assign(new_running_mean)
        assign_running_std = running_std.assign(new_running_std)
        with tf.control_dependencies([assign_running_mean, assign_running_std]):
            self.output_mean = tf.identity(batchMean)
            self.output_std = tf.identity(batchStd)

    def __call__(self):
        return [self.output_mean, self.output_std]
    

    
class UseAvg(object):
    def __init__(self, running_mean, running_std):
        self.output_mean = running_mean
        self.output_std = running_std

    def __call__(self):
        return [self.output_mean, self.output_std]
   
