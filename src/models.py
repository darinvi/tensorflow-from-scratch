from .layers import *
from .tensor import Tensor
from .optimizers import *
from .losses import *
from .metrics import *

class Model:
    def __init__(self):
        self.layers = []
        self.built = False

    def add(self, layer):
        if not isinstance(layer, InputLayer) and layer.input_shape is None and len(self.layers) == 0:
            raise Exception("First layer must be an InputLayer or have an input_shape attribute")

        self.layers.append(layer)

    def build(self):
        out_shape = self.layers[0].output_shape()
        self.layers[0].build()
        for layer in self.layers[1:]:
            layer.build(out_shape)
            out_shape = layer.output_shape()

        self.built = True

    def fit(self, X, y, epochs=10, batch_size=32):
        if not self.built:
            self.build()

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

    def compile(self, optimizer, loss_fn, metrics=[]):
        self._init_optimizer(optimizer)
        self._init_loss_fn(loss_fn)
        self._init_metrics(metrics)

    def _init_optimizer(self, optimizer):
        if isinstance(optimizer, str):
            optimizer = OPTIMIZERS[optimizer]()

        self.optimizer = optimizer

    def _init_loss_fn(self, loss_fn):
        if isinstance(loss_fn, str):
            loss_fn = LOSS_FUNCTIONS[loss_fn]()

        self.loss_fn = loss_fn

    def _init_metrics(self, metrics):
        if not isinstance(metrics, list):
            raise Exception("Metrics must be a list")

        self.metrics = []

        for metric in metrics:
            if isinstance(metric, str):
                metric = METRICS[metric]()
                
            if not isinstance(metric, Metric):
                raise Exception("Metrics should be either a valid metric name or a Metric object")

            self.metrics.append(metric)

class Sequential(Model):
    def __init__(self, layers=[]):
        if not isinstance(layers, list):
            raise Exception("Layers must be a list")
        
        super().__init__()
        for layer in layers:
            self.add(layer)

    


