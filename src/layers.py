from abc import ABC, abstractmethod
from activations import *
from initializers import VarianceScaling

class Layer(ABC):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.built = False

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def build(self, input_shape):
        pass

    def _infer_input_shape(self, X):
        shape = X.shape
        if len(shape) == 1:
            raise Exception("Input must be at least 2D: batch size, features.")
        
        return shape[1:]

    def __call__(self, X):
        if not self.built:
            input_shape = self._infer_input_shape(X)
            self.build(input_shape)
        self._validate_input(X)
        return self.forward(X)

    def _validate_input(self, X):
        # TODO use my own tensor class
        if not isinstance(X, np.ndarray):
            raise Exception("Input must be a numpy array")

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

class Dense(Layer):
    def __init__(self, n_neurons, activation=None, kernel_initializer='glorot_normal'):
        self.activation = activation
        self.n_neurons = n_neurons
        self.kernel_initializer = kernel_initializer

    def get_size(self):
        return self.n_neurons

    def build(self, input_shape):
        self._init_activation()
        self._init_kernel(input_shape)
        self.biases = np.zeros((1, self.n_neurons))
        self.built = True

    def forward(self, inputs):
        output = inputs @ self.weights + self.biases
        return self.activation.forward(output)
