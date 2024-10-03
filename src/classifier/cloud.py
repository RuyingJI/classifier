import numpy as np
import matplotlib.pyplot as plt
class cloud:
    def __init__(self, loc1, scale1, loc2, scale2, size1, size2):
        self.loc1 = loc1  # 类别 1 的均值
        self.scale1 = scale1  # 类别 1 的标准差
        self.loc2 = loc2  # 类别 2 的均值
        self.scale2 = scale2  # 类别 2 的标准差
        self.size1 = size1  # 类别 1 的数据点数量和维度
        self.size2 = size2  # 类别 2 的数据点数量和维度

    def generate_data(self):
        # 生成类别 1 和类别 2 的数据
        class1 = np.random.normal(loc=self.loc1, scale=self.scale1, size=self.size1)
        class2 = np.random.normal(loc=self.loc2, scale=self.scale2, size=self.size2)
        
        # 合并两个类别的数据
        X = np.vstack((class1, class2))  # 将类别 1 和类别 2 的数据垂直合并
        y = np.hstack((np.ones(self.size1[0]), np.ones(self.size2[0]) * 2))  # 生成类别标签
        
        return X, y  # 返回生成的数据和对应的标签
    
    def plot_data(X, y):
        """绘制数据点的函数"""
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k')  # 绘制数据点
        plt.gca().set_aspect('equal')  # 设置坐标轴比例