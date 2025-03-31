import numpy as np

class Tensor:
    def __init__(self, value):
        self.value = np.array(value, dtype=np.float32)
        self.grad = np.zeros_like(self.value)
        self.parents = []
        self.grad_fn = None

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        
        self.grad += grad

        if self.grad_fn:
            self.grad_fn(grad)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value + other.value)

        def grad_fn(grad):
            self.backward(grad)
            other.backward(grad)
        
        out.grad_fn = grad_fn
        out.parents = [self, other]
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value)

        def grad_fn(grad):
            self.backward(grad * other.value)
            other.backward(grad * self.value)
        
        out.grad_fn = grad_fn
        out.parents = [self, other]
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.value, other.value))

        def grad_fn(grad):
            self.backward(np.dot(grad, other.value.T))
            other.backward(np.dot(self.value.T, grad))
        
        out.grad_fn = grad_fn
        out.parents = [self, other]
        return out
    