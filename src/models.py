from layers import Layer
from tensor import Tensor

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, epochs=10, batch_size=32):
        X = Layer._validate_input(X)
        y = Layer._validate_input(y)

        for epoch in range(epochs):
            for i in range(0, len(X.value), batch_size):
                X_batch = Tensor(X.value[i:i+batch_size])
                y_batch = Tensor(y.value[i:i+batch_size])

                out = X_batch
                for layer in self.layers:
                    out = layer(out)

                loss = self.loss_fn.forward(out, y_batch)

                loss.backward()
                self.optimizer.step(self.layers)

class Sequential(Model):
    def __init__(self, layers=[]):
        if not isinstance(layers, list):
            raise Exception("Layers must be a list")
        
        self.layers = layers
    


