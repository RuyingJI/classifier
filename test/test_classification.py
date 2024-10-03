import numpy as np
import sys 
# 获取工作目录，即test目录的上一级目录
workingDir= __file__.split( "test/" )[0]
#print(__file__)
#print(workingDir)
# 将工作目录添加到sys.path，使Python能从这个目录中查找模块
sys.path.append(workingDir)
from src.classifier.classifierSimpleY import ClassifierSimpleY
from src.classifier.classifierCircle import ClassifierCircle
from src.classifier.classifierLinear import ClassifierLinear

from src.classifier.cloud import cloud
def test_classification():

    print("Testing SimpleY classifier...")
   # np.random.seed(42)
    """	X1 和 X2 分别是两类数据，X1 的数据中心在 (20, 30)，X2 的数据中心在 (30, 10)。每类有 100 个数据点。
    X1 = np.random.randn(100, 2) + [20, 30]  # 类别 1 数据，均值为 (20, 30)
    X2 = np.random.randn(100, 2) + [30, 10]  # 类别 2 数据，均值为 (30, 10)
    X = np.vstack((X1, X2))  # 将两类数据点合并在一起，形成数据集
    y = np.hstack((np.ones(100), np.ones(100) * 2))  # 类别标签，前 100 个为 Class 1，后 100 个为 Class 2"""
    loc1 = [20, 30]  # class 1 的均值 (x, y)
    scale1 = [5, 5]  # class 1 的标准差 (x, y)
    loc2 = [30, 10]  # class 2 的均值 (x, y)
    scale2 = [5, 5]  # class 2 的标准差 (x, y)
    size1 = (100, 2)  # class 1 生成 100 pts，(x,y)
    size2 = (100, 2)  # class 2 生成 100 pts，(x,y)
    data_generator = cloud(loc1, scale1, loc2, scale2, size1, size2)
    X, y = data_generator.generate_data()


    # ClassifierSimpleY
    simpleY = ClassifierSimpleY()  
    simpleY.optimizeParameters(X,y)
    simpleY.plot(X, y)  

    # ClassifierCircle
    circleClassifier = ClassifierCircle()  
    circleClassifier.optimizeParameters(X, y)  
    circleClassifier.plot(X, y) 

    #ClassifierLinear
    linearClassifier = ClassifierLinear() 
    linearClassifier.optimizeParameters(X, y)  
    
    linearClassifier.optimizeParameters(X, y ,0.001,linearClassifier.getParameters()[0]-0.1, linearClassifier.getParameters()[0]+0.1)
    linearClassifier.plot(X, y)  

test_classification()

