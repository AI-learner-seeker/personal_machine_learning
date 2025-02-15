import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 生成示例数据
X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 数据标准化
X = StandardScaler().fit_transform(X)

# 使用DBSCAN进行聚类
db = DBSCAN(eps=0.3, min_samples=5)
db.fit(X)

# 获取聚类标签
labels = db.labels_

# 可视化结果
plt.figure(figsize=(8, 6))

# 绘制核心样本
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# 绘制不同簇的点
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点用黑色表示
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    # 绘制核心点
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=50, c=[col], marker='o', edgecolors='k')

    # 绘制非核心点
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 0], xy[:, 1], s=30, c=[col], marker='o', edgecolors='k')

plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()