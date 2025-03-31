class VarianceScaling:
    def __init__(self, scale=1.0, mode='fan_in', distribution='normal', seed=None):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution
        self.seed = seed

    def get_weights(self, input_shape, output_shape):
        return
    