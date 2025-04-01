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
        if isinstance(input_shape, tuple) and len(input_shape) == 1:
            input_shape = input_shape[0]

        if isinstance(output_shape, tuple) and len(output_shape) == 1:
            output_shape = output_shape[0]

        if isinstance(input_shape, int) and isinstance(output_shape, int):
            # return Tensor(np.random.randn(input_shape, output_shape))
            return Tensor.randn(input_shape, output_shape)

        # FIXME more complex building of weights based on shapes