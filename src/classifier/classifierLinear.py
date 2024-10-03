from . import Classifier # 先from
from .classifierSimpleY  import ClassifierSimpleY # 先from
import numpy as np
import matplotlib.pyplot as plt
from .cloud import cloud

class ClassifierLinear(ClassifierSimpleY):
    def __init__(self, pente=1.0, intercept=0.0):
        super().__init__()
        self.setParameters([pente, intercept])  # 设置斜率和截距参数
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
        plt.plot(x_values, y_values, 'r--')  # 分类线用红色虚线表示

        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.savefig("classifierLinear.png")  # 保存图像为 PNG 文件
        plt.clf()

    def optimizeParameters(self, X, y, precision=0.1,pente1=-5,pente2=5):  # 通过穷举搜索优化斜率和截距
      #  print(f"Starting linear optimization with precision: {precision}")
        
        best_params = self.getParameters()  # 获取当前参数
        min_error = float('inf')  # 初始化最小错误数为无穷大

        # 设置斜率和截距的搜索范围
        pente_range = np.arange(pente1, pente2, precision)  # 可以调整范围
        intercept_range = np.arange(X[:, 1].min(), X[:, 1].max(), precision)

        # 遍历所有可能的斜率和截距组合，找到最优解
        for pente in pente_range:
            for intercept in intercept_range:
                self.setParameters([pente, intercept])  # 设置当前斜率和截距
                error = self.countError(X, y)  # 计算当前参数下的分类错误数

                # 打印当前的斜率、截距和错误数
               # print(f"Testing pente: {pente:.2f}, intercept: {intercept:.2f}, Error count: {error}")

                if error < min_error:  # 如果错误数减少，更新最优参数
                    min_error = error
                    best_params = [pente, intercept]
                 #   print(f"New best parameters found: pente = {pente:.2f}, intercept = {intercept:.2f}, Error = {min_error}")
        # 设置最优参数
        self.setParameters(best_params)
        print(f" Best pente de sys lineaire: {best_params[0]}, Best intercept: {best_params[1]}, Min error: {min_error}")
        return best_params