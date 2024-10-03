from . import Classifier # 先from
import numpy as np
import matplotlib.pyplot as plt
from .cloud import cloud
class ClassifierSimpleY(Classifier):
    def __init__(self, y_threshold=0.0):
        super().__init__()                  #----> 继承
        """
        初始化分类器时，接受一个 y_threshold 参数，表示用于分类的阈值。默认值为 0.0。将该阈值存储在 params 中。
        """
        self.setParameters([y_threshold])


    def predict(self, X):
        # 输入：X 是一个二维数组，第一列是 X 坐标，第二列是 Y 坐标。
        y_threshold = self.getParameters()[0]
        """逻辑：------> np.where
            通过检查每个数据点的 Y 坐标是否大于 y_threshold，
            大于的分为 Class-2（返回 2），小于等于的分为 Class-1（返回 1）。
            这个逻辑由 np.where(X[:, 1] > y_threshold, 2, 1) 完成。
        """
        return np.where(X[:, 1] > y_threshold, 1, 2)

    def plot(self, X, y):
        y_threshold = self.getParameters()[0]
        cloud.plot_data(X, y)   
        plt.axhline(y=y_threshold, color='r', linestyle='--', label=f"y = {y_threshold}")
        """绘制二维数据集，并画出分类阈值 y_threshold。
            数据点通过颜色区分，分类线通过 plt.axhline 画出，颜色为红色，样式为虚线。
            ？？？？？
            plt.axhline(y=0.5) 会在 y = 0.5 的位置绘制一条水平线。axhline 通常用于图形中添加参考线，例如展示阈值、均值或某个重要的参考点。
        """
        plt.legend()
        plt.savefig("classifierSimpleY.png")                                    #
        plt.clf()                    

    def optimizeParameters(self, X, y, precision=0.1):

        """	
        •	步骤：
	1.	先确定数据集的 Y 坐标的最小值和最大值（y_min 和 y_max）。
	2.	然后遍历不同的 y_threshold 值（从 y_min 到 y_max），计算在每个阈值下的分类错误数。
	3.	找到分类错误最小的 y_threshold 并返回。
        """
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        best_threshold = (y_min+y_max)/2
        min_error = float('inf')
       # print(f"y_min: {y_min}, y_max: {y_max}, precision: {precision}")

        """     ???
            float('inf') 表示正无穷大。这是一个特殊的浮点数，表示的值比任何其他数都大。它通常用于在算法中初始化最小值比较变量。"""

        for y_threshold in np.arange(y_min,y_max, precision):
           # print(f"Testing y_threshold: {y_threshold}")
            self.setParameters([y_threshold])
            error = self.countError(X, y)
           # print(f"Error count for y_threshold {y_threshold}: {error}")
            if error < min_error:
                min_error = error
                best_threshold = y_threshold

        for y_threshold in np.arange(best_threshold-0.1 ,best_threshold+0.1, 0.001):
            
            self.setParameters([y_threshold])
            error = self.countError(X, y)
            if error < min_error:
                min_error = error
                best_threshold = y_threshold

        self.setParameters([best_threshold])
        print(f"best_threshold de SimpleY: {best_threshold}, Min Error count: {min_error}")
        return best_threshold