from . import Classifier # 先from
import numpy as np
import matplotlib.pyplot as plt
from .cloud import cloud


class ClassifierCircle(Classifier):
    def __init__(self, center=(0.0, 0.0), radius=1.0):
        super().__init__()
        self.setParameters([center[0], center[1], radius])
    """初始化圆形分类器，默认圆心为 (0.0, 0.0)，半径为 1.0。将圆心坐标和半径存储为参数"""

    def predict(self, X):
        x_center, y_center, radius = self.getParameters()
        distances = np.sqrt((X[:, 0] - x_center) ** 2 + (X[:, 1] - y_center) ** 2)

        return np.where(distances <= radius, 1, 2)
    """逻辑：计算每个点到圆心的欧氏距离 distances，如果距离小于等于半径，则该点属于 Class-1，否则属于 Class-2。"""

    def plot(self, X, y):
        x_center, y_center, radius = self.getParameters()
       # plt.gca().set_aspect('equal')
        cloud.plot_data(X, y)   
        circle = plt.Circle((x_center, y_center), radius, color='r', fill=False, linestyle='--')    #-->创建圆形边界，并添加到图形中。
        plt.gca().add_artist(circle)
        """，gca().add_artist(circle) 方法将圆添加到当前的绘图区。"""
        plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        plt.savefig("classifierCircle.png")                                    # 保存图像为 PNG 文件
        plt.clf()     
    
    
    def optimizeParameters(self, X, y, precision=0.1):
        # 1. 找到 Class-1 的数据点
        #class1_points = X[y == 1]
        
        # 2. 计算 Class-1 数据点的中心作为圆心
        #x_center, y_center = class1_points.mean(axis=0)
        x_center, y_center = X[:, 0].min(),X[:, 1].max()  
       # print(f"Class-1 Center: ({x_center}, {y_center})")

        # 3. 设置圆心为 Class-1 数据的中心
        best_params = [x_center, y_center, 0.1]  # 初始参数，圆心是 class1 的中心，半径从 0.1 开始
        min_error = float('inf')  # 初始化最小错误数为无穷大
       # print("Starting radius optimization...")
        # 4. 仅优化半径参数
        x_min, x_max = X[:, 0].min(), X[:, 0].max()  # X 坐标范围
        y_min, y_max = X[:, 1].min(), X[:, 1].max()  # Y 坐标范围

        # 打印范围
       # print(f"Data X range: [{x_min}, {x_max}], Y range: [{y_min}, {y_max}]")

        for radius in np.arange(0.1, max(x_max - x_min, y_max - y_min), precision):
            self.setParameters([x_center, y_center, radius])  # 设置当前圆心和半径
            error = self.countError(X, y)  # 计算错误数
            # 打印当前半径和错误数
             #    print(f"Testing radius: {radius:.2f}, Error count: {error}")

            if error < min_error:  # 如果错误数更小，更新最优参数
                min_error = error
                best_params = [x_center, y_center, radius]
            #   print(f"New best radius found: {radius:.2f} with error count: {min_error}")
                
        for radius in np.arange(best_params[2]-0.1, best_params[2]+0.1, 0.0001):
            self.setParameters([x_center, y_center, radius])  # 设置当前圆心和半径
            error = self.countError(X, y)  # 计算错误数
            # 打印当前半径和错误数
           # print(f"Testing radius: {radius:.5f}, Error count: {error}")

            if error < min_error:  # 如果错误数更小，更新最优参数
                min_error = error
                best_params = [x_center, y_center, radius]
             #   print(f"New best radius found: {radius:.5f} with error count: {min_error}")
                    

        # 5. 设置最优参数
        self.setParameters(best_params)
        print(f"Best radius de sys circle: {best_params[2]}, Min error: {min_error}")
        return best_params