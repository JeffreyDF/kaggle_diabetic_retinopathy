import lasagne as nn
from lasagne import nonlinearities

try:
    from theano.sandbox.cuda import dnn
    from lasagne.layers.dnn import DNNLayer, conv_output_length
except ImportError:
    print "No cudnn, not imported."
    DNNLayer = nn.layers.Layer


class ApplyNonlinearity(nn.layers.Layer):

    def __init__(self, input_layer, nonlinearity=nonlinearities.softmax):
        super(ApplyNonlinearity, self).__init__(input_layer)
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, *args, **kwargs):
        return self.nonlinearity(input)
