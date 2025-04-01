from abc import ABC, abstractmethod
from .tensor import Tensor

class Activation(ABC):
    def __call__(self, X):
        return self.call(X)
    
    def forward(self, X):
        return self.call(X)

class LinearActivation(Activation):
    def call(self, X):
        return X

class ReLU(Activation):
    def call(self, X):
        return Tensor(X).maximum(0)
    
ACTIVATIONS = {
    'linear': LinearActivation,
    'relu': ReLU,
}