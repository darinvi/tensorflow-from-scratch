import numpy as np
from abc import ABC, abstractmethod
from .tensor import Tensor

class Metric(ABC):
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass
    
    def _validate_and_extract_np_array(self, y_pred, y_true):
        if isinstance(y_pred, Tensor):
            y_pred = y_pred.value
        
        if isinstance(y_true, Tensor):
            y_true = y_true.value

        if not isinstance(y_pred, np.ndarray) or not isinstance(y_true, np.ndarray):
            raise ValueError("y_pred and y_true must be numpy arrays")

        return y_pred, y_true

class Accuracy(Metric):
    def forward(self, y_pred, y_true):
        y_pred, y_true = self._validate_and_extract_np_array(y_pred, y_true)
        return np.mean(y_pred == y_true)

class RMSE(Metric):
    def forward(self, y_pred, y_true):
        y_pred, y_true = self._validate_and_extract_np_array(y_pred, y_true)
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

METRICS = {
    'accuracy': Accuracy,
    'rmse': RMSE
}
