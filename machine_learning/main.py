#PCA算法
#自带的python编辑器，未安装其他第三方库
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def PCA_example():
    #加载鸢尾花数据集
    data = load_iris()
    print(data)
    y = data.target
    print(y)
    x = data.data
    print(x)
    n_components = 2
    #pca降维过程
    pca = PCA(n_components = n_components)
    reduced_x = pca.fit_transform(x)
    print(reduced_x)
    #可视化降维结果
    red_x,red_y = [],[]
    blue_x,blue_y = [],[]
    green_x,green_y = [],[]
    for i in range(len(reduced_x)):
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])

    plt.scatter(red_x,red_y,c = 'r', marker = 'x', label = 'Class1')
    plt.scatter(blue_x,blue_y,c = 'b', marker = 'D', label = 'Class2')
    plt.scatter(green_x,green_y,c = 'g', marker = '.', label = 'Class3')
    plt.legend()
    plt.show()

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    PCA_example()


