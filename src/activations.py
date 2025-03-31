from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    def __call__(self, X):
        return self.call(X)
    
    def forward(self, X):
        return self.call(X)

    @abstractmethod
    def call(self, X):
        pass

class LinearActivation(Activation):
    def call(self, X):
        return X

class ReLU(Activation):
    def call(self, X):
        return np.maximum(X, 0)

    def backward(self, X):
        return np.where(X > 0, 1, 0)

ACTIVATIONS = {
    'linear': LinearActivation,
    'relu': ReLU,
}