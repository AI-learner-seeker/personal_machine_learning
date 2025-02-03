# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

# 加载鸢尾花数据集
iris = load_iris()
print("iris是：\n",iris)
X = iris.data  # 特征
print("iris特征X是：\n",X)
y = iris.target  # 标签
print("iris标签y是：\n",y)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化，均值为零，标准差为一的分布
scaler = StandardScaler()#创建了一个StandardScaler的实例，命名为scaler
#使用fit_transform方法对训练数据集X_train进行拟合和转换。fit部分计算了数据的均值和标准差，而transform部分则使用这些参数将数据转换为标准化形式。这一步的结果是将X_train中的每个特征都转换为了均值为0、标准差为1的分布。
X_train = scaler.fit_transform(X_train)
#对于测试数据集X_test，只使用transform方法进行转换。这是因为测试数据应该使用训练数据计算得到的均值和标准差进行标准化，以保持训练和测试数据的一致性。如果再次使用fit_transform，则测试数据将基于其自身的统计量进行标准化，这可能导致数据泄露问题，影响模型的泛化能力。
X_test = scaler.transform(X_test)

# 使用PCA将数据降维到2维（方便可视化）
pca = PCA(n_components=2)#创建了一个PCA的实例，命名为pca，并指定了要将数据降维到的维度数n_components=2
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 创建KNN分类器
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train_pca, y_train)

# 预测测试集
y_pred = knn.predict(X_test_pca)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 打印分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 可视化分类结果和决策边界

def plot_decision_boundary(X, y, classifier, resolution=0.02):
    # 定义颜色和标记
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 绘制决策边界
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 绘制样本点
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=iris.target_names[cl], edgecolor='black')

    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')
    plt.title('KNN Classification on Iris Dataset (PCA-reduced)')
    plt.legend(loc='upper right')
    plt.show()

# 绘制训练集的决策边界和分类结果
plot_decision_boundary(X_train_pca, y_train, classifier=knn)

# 绘制测试集的分类结果
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap=ListedColormap(('red', 'blue', 'lightgreen')), edgecolor='black', s=100)
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('KNN Predictions on Test Set (PCA-reduced)')
plt.show()