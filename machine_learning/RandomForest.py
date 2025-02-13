import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练随机森林回归模型
rf = RandomForestRegressor(
    n_estimators=200,  # 增加树的数量
    max_depth=15,       # 限制树的最大深度
    min_samples_split=5,  # 分割内部节点所需的最小样本数
    random_state=42,
    n_jobs=-1          # 使用所有可用的CPU核心
)
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差(MSE): {mse:.4f}")
print(f"决定系数(R²): {r2:.4f}")

# 可视化结果
plt.figure(figsize=(15, 5))

# 1. 实际值 vs 预测值散点图
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

# 2. 特征重要性条形图
plt.subplot(1, 3, 2)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.barh(range(X.shape[1]), importances[indices], align='center')
plt.yticks(range(X.shape[1]), X.columns[indices])
plt.xlabel('Feature Importance')
plt.title('Feature Importance Ranking')

# 3. 残差分布图
plt.subplot(1, 3, 3)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='k', linestyles='dashed')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')

plt.tight_layout()
plt.show()