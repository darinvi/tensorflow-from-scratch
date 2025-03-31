class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "trainable") or layer.trainable is False:
                continue
            self._step(layer)
            
    @staticmethod
    def _step(self, layer):
        ...

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def _step(self, layer):
        layer.weights.value -= self.lr * layer.weights.grad
        layer.biases.value -= self.lr * layer.biases.grad