import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

class Accuracy(Metric):
    def forward(self, y_pred, y_true):
        return np.mean(y_pred == y_true)

METRICS = {
    'accuracy': Accuracy
}
