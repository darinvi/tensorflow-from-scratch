import numpy as np
from .tensor import Tensor
class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    # FIXME return based on parameters isntead of random like that..
    def get_weights(self, input_shape, output_shape):
        if isinstance(input_shape, tuple):
            input_size = input_shape[0]
        else:
            input_size = input_shape
            
        if isinstance(output_shape, tuple):
            output_size = output_shape[0]
        else:
            output_size = output_shape
            
        return Tensor(np.random.randn(input_size, output_size))

        # FIXME more complex building of weights based on shapes