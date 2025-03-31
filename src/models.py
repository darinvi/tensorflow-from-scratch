class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y):
        pass

class Sequential(Model):
    def __init__(self, layers=[]):
        if not isinstance(layers, list):
            raise Exception("Layers must be a list")
        
        self.layers = layers
    

