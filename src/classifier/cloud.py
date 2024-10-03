import numpy as np
import matplotlib.pyplot as plt

class cloud:
    def __init__(self, loc1, scale1, loc2, scale2, size1, size2):
        self.loc1 = loc1  
        self.scale1 = scale1  
        self.loc2 = loc2  
        self.scale2 = scale2  
        self.size1 = size1 
        self.size2 = size2  

    def generate_data(self):
        class1 = np.random.normal(loc=self.loc1, scale=self.scale1, size=self.size1)
        class2 = np.random.normal(loc=self.loc2, scale=self.scale2, size=self.size2)
        

        X = np.vstack((class1, class2))  
        y = np.hstack((np.ones(self.size1[0]), np.ones(self.size2[0]) * 2)) 
        
        return X, y 
    
    def plot_data(X, y):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')  
        plt.gca().set_aspect('equal')  