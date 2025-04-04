from abc import ABC, abstractmethod

class Loss(ABC):
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

class MSE(Loss):
    def forward(self, y_pred, y_true):
        return (y_pred - y_true) ** 2

LOSS_FUNCTIONS = {
    'mse': MSE,
}