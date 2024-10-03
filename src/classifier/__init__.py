import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self):

        self.params = []

    def getParameters(self):
        return self.params

    def setParameters(self, params):
        self.params = params
        
    def countError(self, X, y):
        predictions = self.predict(X)
        return np.sum(predictions != y)
    
    
    def optimizeParameters(self, X, y, precision=0.01):
        raise NotImplementedError("This method should be implemented by subclass.")

    def predict(self, X):
        raise NotImplementedError("This method should be implemented by subclass.")

    def plot(self, X, y):
        raise NotImplementedError("This method should be implemented by subclass.")