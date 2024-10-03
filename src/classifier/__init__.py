import numpy as np
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self):
        # 初始化分类器的参数
        self.params = []

    def getParameters(self):
        # 获取分类器的参数;     分别用于获取和设置分类器的参数。每个分类器有不同的参数，比如简单 Y 分类器的阈值，圆形分类器的圆心和半径等。
        return self.params

    def setParameters(self, params):
        # 设置分类器的参数      分别用于获取和设置分类器的参数
        self.params = params
        
#对于每个classifier是相同的方法。
    def countError(self, X, y):
        # 计算分类错误数        计算分类器在数据集上的错误分类数。predict 方法返回每个数据点的预测类别，然后将预测类别与实际标签 y 进行比较，
        predictions = self.predict(X)
        return np.sum(predictions != y)
    
    
#需要在子类中进行具体实现。这个设计符合面向对象编程的多态性原则
    def optimizeParameters(self, X, y, precision=0.01):
        # 参数优化函数，需在子类中实现
        raise NotImplementedError("This method should be implemented by subclass.")

    def predict(self, X):
        # 预测函数，需在子类中实现
        raise NotImplementedError("This method should be implemented by subclass.")

    def plot(self, X, y):
        # 绘制分类结果，需在子类中实现
        raise NotImplementedError("This method should be implemented by subclass.")