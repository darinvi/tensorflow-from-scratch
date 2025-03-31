import numpy as np

class Tensor:
    def __init__(self, input):
        self.value = self._init_input(input)
        self.grad = np.zeros_like(self.value, dtype=np.float32)
        self.parents = [] # not directly used
        self.grad_fn = None

    # propagate data validation to np.array's constructor
    @staticmethod
    def _init_input(input):
        if not isinstance(input, np.ndarray):
            input = np.array(input, dtype=np.float32)
        
        if not input.dtype == np.float32:
            input = input.astype(np.float32)

        return input

    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.value)
        
        self.grad += grad

        if self.grad_fn:
            self.grad_fn(grad)

    def _validate_input(func):
        def wrapper(self, other):
            other = other if isinstance(other, Tensor) else Tensor(other)
            return func(self, other)
        return wrapper

    def _return_tensor(self, other, out, grad_fn):
        out.grad_fn = grad_fn
        out.parents = [self, other]
        return out

    @_validate_input
    def __add__(self, other):
        out = Tensor(self.value + other.value)

        def grad_fn(grad):
            self.backward(grad)
            other.backward(grad)
        
        return self._return_tensor(other, out, grad_fn)

    @_validate_input
    def __mul__(self, other):
        out = Tensor(self.value * other.value)

        def grad_fn(grad):
            self.backward(grad * other.value)
            other.backward(grad * self.value)
        
        return self._return_tensor(other, out, grad_fn)

    @_validate_input
    def matmul(self, other):
        out = Tensor(np.dot(self.value, other.value))

        def grad_fn(grad):
            self.backward(np.dot(grad, other.value.T))
            other.backward(np.dot(self.value.T, grad))
        
        return self._return_tensor(other, out, grad_fn)
    
    def __repr__(self):
        return f"Tensor(value={self.value}, grad={self.grad})"

