import lasagne as nn
from lasagne.layers import dnn

class Conv2DLayer(dnn.Conv2DDNNLayer):

    def __init__(self, incoming, learning_rate_scale=1.0,
                 **kwargs):
        super(Conv2DLayer, self).__init__(incoming=incoming, **kwargs)
        self.learning_rate_scale = learning_rate_scale


class DenseLayer(nn.layers.DenseLayer):

    def __init__(self, incoming, learning_rate_scale=1.0,
                 **kwargs):
        super(DenseLayer, self).__init__(incoming=incoming, **kwargs)
        self.learning_rate_scale = learning_rate_scale
