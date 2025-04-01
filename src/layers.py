from abc import ABC, abstractmethod
from .activations import *
from .initializers import VarianceScaling
from .tensor import Tensor

class Layer(ABC):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.built = False

    @abstractmethod
    def forward(self, X):
        ...

    @abstractmethod
    def output_shape(self):
        ...

    def _infer_input_shape(self, X):
        shape = X.shape
        if len(shape) == 1:
            raise Exception("Input must be at least 2D: batch size, features.")
        
        return shape[1:]

    @abstractmethod
    def build(self, input_shape=None):
        ...

    def __call__(self, X):
        if not self.built:
            input_shape = self._infer_input_shape(X)
            self.build(input_shape)
        self._validate_input(X)
        return self.forward(X)

    @staticmethod
    def _validate_input(X):
        if not isinstance(X, Tensor):
            X = Tensor(X)
        return X

class TrainableLayer(Layer):
    def __init__(self, input_shape=None):
        super().__init__(input_shape)
        self.trainable = True

    def _init_activation(self):
        if isinstance(self.activation, Activation):
            return
    
        if (act := ACTIVATIONS.get(self.activation)) is None:
            raise Exception(f"Invalid activation function: {self.activation}. Valid options are: {ACTIVATIONS.keys()}")
        
        self.activation = act()

    def _init_kernel(self, input_shape):
        if not isinstance(self.kernel_initializer, VarianceScaling):
            method, distribution = self.kernel_initializer.split('_')
            method = {'he': 'fan_in', 'glorot': 'fan_avg'}.get(method)
            if method is None:
                raise Exception(f"If initializing kernel with a string, method should be 'he' or 'glorot'")
            
            self.kernel_initializer = VarianceScaling(mode=method, distribution=distribution)
        
        self.weights = self.kernel_initializer.get_weights(input_shape, self.n_neurons)

class InputLayer(Layer):
    def __init__(self, input_shape=None):
        if input_shape is not None:
            raise Exception("InputLayer requires input_shape at initialization time")
        super().__init__()

    def output_shape(self):
        return self.input_shape
    
    def build(self, input_shape=None):
        self.built = True

class Dense(TrainableLayer):
    def __init__(self, n_neurons, activation='linear', kernel_initializer='glorot_normal', input_shape=None):
        super().__init__(input_shape)
        self.activation = activation
        self.n_neurons = n_neurons
        self.kernel_initializer = kernel_initializer

    def output_shape(self):
        return (self.n_neurons,)

    def build(self, input_shape=None):
        if input_shape is None and self.input_shape is None:
            raise Exception("Input shape must be provided either at build time or at initialization time")

        if input_shape and self.input_shape and input_shape != self.input_shape:
            raise Exception("Input shape must be provided either at build time or at initialization time")

        shape = input_shape or self.input_shape
        self.input_shape = shape
        self._init_activation()
        self._init_kernel(shape)
        self.biases = Tensor.zeros((1, self.n_neurons))
        self.built = True

    def forward(self, inputs):
        output = inputs.matmul(self.weights) + self.biases
        return self.activation.forward(output)
