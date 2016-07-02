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
        if mode in [0,2]:
            assert axis > 0, "specify axis>0 (channel dim) for feature maps"
            assert axis < input_shape, "axis to large"
            depth = input_shape[axis]
            reduction_indices = [idx for idx in range(len(input_shape)) if idx != axis]
        elif mode in [1,3]:
            assert len(input_shape)==2, "sample wise mode=0 is for flattened 2D input tensor, not convnets"
            depth = input_shape[1]
            reduction_indices = [0]

        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]), trainable=True)
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]), trainable=True)

        self.running_mean = tf.Variable(tf.zeros(depth), trainable=False, name='running_mean') 
        self.running_std = tf.Variable(tf.ones(depth), trainable=False, name='running_std') 
  
        if mode in [0,2]:
            featuremapBatch = FeaturemapBatch(inputTensor,
                                              self.beta,
                                              self.gamma)
        if mode == 0:
            featuremapMovingAvg = FeaturemapMovingAvg(inputTensor,
                                                      self.beta,
                                                      self.gamma)
        if mode in [1,3]:
            sampleBatch = SampleBatch(inputTensor,
                                      self.beta,
                                      self.gamma)
        if mode == 3:
            samplewiseMovingAvg = SampleMovingAvg(inputTensor,
                                                  self.beta,
                                                  self.gamma)
            
        ### create ops
        if mode == 0:
            batchMean, stdNorm = tf.cond(train_placeholder, 
                                         featuremapBatch, 
                                         featuremapMovingAvg)
        elif mode == 1:
            batchMean, stdNorm = sampleBatch()
        elif mode == 2:
            batchMean, stdNorm = featuremapBatch()
        elif mode == 3:
            batchMean, stdNorm = tf.cond(train_placeholder,
                                         samplewiseBatch,
                                         samplewiseMovingAvg)

        meanNorm = tf.sub(inputTensor, batchMean)
        stdNorm = tf.div(meanNorm, stdNorm)
        self.op = stdNorm

    def getOp(self):
        return self.op

class FeaturemapBatch(object):
    def __init__(self, xx, beta, gama):
        self.op = tf.identity(xx)

    def __call__(self):
        return [self.op, self.op]

class FeaturemapMovingAvg(object):
    def __init__(self, xx, beta, gama):
        self.op = tf.identity(xx)

    def __call__(self):
        return [self.op, self.op]

class SampleBatch(object):
    def __init__(self, xx, beta, gama):
        self.op = tf.identity(xx)

    def __call__(self):
        return [self.op, self.op]

class SampleMovingAvg(object):
    def __init__(self, xx, beta, gama):
        self.op = tf.identity(xx)

    def __call__(self):
        return [self.op, self.op]

    
