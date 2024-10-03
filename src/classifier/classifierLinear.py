from . import Classifier 
from .classifierSimpleY  import ClassifierSimpleY 
import numpy as np
import matplotlib.pyplot as plt
from .cloud import cloud

class ClassifierLinear(ClassifierSimpleY):
    def __init__(self, pente=1.0, intercept=0.0):
        super().__init__()
        self.setParameters([pente, intercept]) 
    """初始化线性分类器，默认斜率为 1.0，截距为 0.0"""

    def predict(self, X):
        pente, intercept = self.getParameters()
        # 对于每个数据点，计算它相对于直线的位置 y = pente * x + intercept
        predictions = np.where(X[:, 1] >= pente * X[:, 0] + intercept, 1, 2)
        return predictions
    """根据斜率和截距计算预测类别。如果点在直线以上，属于 Class-1，否则属于 Class-2"""

    def plot(self, X, y):
        pente, intercept = self.getParameters()
        cloud.plot_data(X, y)   

        # 绘制线性分类边界 y = pente * x + intercept
        x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_values = pente * x_values + intercept
        plt.plot(x_values, y_values, 'r--') 

        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.savefig("classifierLinear.png")  
        plt.clf()

    def optimizeParameters(self, X, y, precision=0.1,pente1=-5,pente2=5):  
      #  print(f"Starting linear optimization with precision: {precision}")
        
        best_params = self.getParameters()  
        min_error = float('inf')  

        pente_range = np.arange(pente1, pente2, precision)  
        intercept_range = np.arange(X[:, 1].min(), X[:, 1].max(), precision)

        for pente in pente_range:
            for intercept in intercept_range:
                self.setParameters([pente, intercept])  
                error = self.countError(X, y) 
               # print(f"Testing pente: {pente:.2f}, intercept: {intercept:.2f}, Error count: {error}")

                if error < min_error: 
                    min_error = error
                    best_params = [pente, intercept]
                 #   print(f"New best parameters found: pente = {pente:.2f}, intercept = {intercept:.2f}, Error = {min_error}")
        self.setParameters(best_params)
        print(f" Best pente de sys lineaire: {best_params[0]}, Best intercept: {best_params[1]}, Min error: {min_error}")
        return best_params