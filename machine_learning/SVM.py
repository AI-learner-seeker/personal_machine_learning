"""SVM支持向量机，有较为复杂的数学公式理论推导，这里重点在于简单介绍一下支持向量机的工作原理，并给出常见的代码（基于SVM的异或数据集划分），对于具体数学细节的推导为我们并不付出过多精力关注"""
#支持向量机（support vector machine）SVM是一类按监督学习方式对数据进行二元分类的广义线性分类器
#距离超平面最近的这几个训练样本点被称为"支持向量"
#两个异类支持向量到超平面的距离之和为”间隔“
#我们求解的目标就是要找到具有"最大间隔"的一个划分超平面，用于二分类任务
#一开始为我们是假设的针对线性可分的数据，训练样本在样本空间或特征空间中是线性可分的，即存在一个超平面能将不同类的样本完全划分开
#但是现实情况中的数据往往线性不可分，如本程序的”异或“数据集的划分，不具有线性可分性，因此就需要进行进一步改进
#可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分. 例如可将原始的二维空间映射到一个合适的三维空间，就能找到一个合适的划分超平面

#（重要！！！！！）如果原始空间是有限维，即属性数有限，那么一定存在一个高维特征空间使样本可分

#所以，这个时候我们要借助一个”核函数“将特征映射到高维度，从而进行分类操作
#核函数有三种常见的，线性核函数、高斯核函数、多项式核函数
#线性核函数主要用于数据线性可分，通常是首先尝试用线性核函数进行分类
#高斯核函数是一种局部性较强的核函数，可以将一个样本映射到一个更高维的空间内，该核函数是应用最广泛的一个，无论样本大小均有较好的性能，且相对于多项式核函数的参数要少

#（重要！！！！！）大多数情况下不知道选什么核函数时，优先使用高斯核函数

#多项式核函数非常适合于正交归一化后的数据，可以实现将低纬的输入空间映射到高维的特征空间，但是多项式核函数的参数多，当多项式阶数较高时，核矩阵的元素值将趋于无穷大或者无穷小，计算复杂度非常大

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

def plot_decision_regions(x,y,model,resolution=0.2):
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max = x[:,0].min() - 1,x[:,0].max() + 1
    x2_min,x2_max = x[:,1].min() - 1,x[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z = model.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor == 1,0],X_xor[y_xor == 1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor == -1,0],X_xor[y_xor == -1,1],c='r',marker='s',label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

svm = SVC(kernel="rbf",random_state=0,gamma=0.1,C=1.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor,svm)


























