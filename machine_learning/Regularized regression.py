# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文

# 加载糖尿病数据集
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化（正则化对特征尺度敏感，必须进行标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 初始化不同正则化模型
ridge = Ridge(alpha=1.0)       # 岭回归（L2正则化）
lasso = Lasso(alpha=0.1)       # Lasso回归（L1正则化）
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 弹性网络

# 训练模型
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

# 预测结果
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)
y_pred_elastic = elastic.predict(X_test)

# 评估模型
def evaluate_model(name, y_true, y_pred):
    print(f"{name}模型评估:")
    print(f"均方误差(MSE): {mean_squared_error(y_true, y_pred):.2f}")
    print(f"判定系数(R²): {r2_score(y_true, y_pred):.2f}\n")

evaluate_model("岭回归", y_test, y_pred_ridge)
evaluate_model("Lasso回归", y_test, y_pred_lasso)
evaluate_model("弹性网络", y_test, y_pred_elastic)

# 可视化系数比较
plt.figure(figsize=(10, 6))
features = diabetes.feature_names

# 绘制系数大小
plt.plot(ridge.coef_, 'o', label="岭回归")
plt.plot(lasso.coef_, 's', label="Lasso回归")
plt.plot(elastic.coef_, '^', label="弹性网络")

plt.xticks(range(len(features)), features, rotation=45)
plt.xlabel("特征索引")
plt.ylabel("系数大小")
plt.title("不同正则化方法的系数比较")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 在原有代码基础上新增以下可视化部分

plt.figure(figsize=(12, 8))

# 生成样本索引（用于x轴）
sample_indices = np.arange(len(y_test))

# 绘制真实值
plt.scatter(sample_indices, y_test, color='black', alpha=0.6, marker='*', s=100, label='真实值')

# 绘制各模型预测值
plt.scatter(sample_indices, y_pred_ridge,  color='red',    alpha=0.5, marker='o', s=60, label='岭回归预测')
plt.scatter(sample_indices, y_pred_lasso,  color='blue',   alpha=0.5, marker='s', s=60, label='Lasso预测')
plt.scatter(sample_indices, y_pred_elastic,color='green', alpha=0.5, marker='^', s=60, label='弹性网络预测')

# 添加辅助线
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         color='grey', linestyle='--', lw=2, label='完美预测线')

# 图表装饰
plt.title("模型预测结果对比（测试集）", fontsize=14)
plt.xlabel("样本索引", fontsize=12)
plt.ylabel("目标值", fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 将图例移到图表外侧
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 添加评估指标文本
text_str = '\n'.join([
    f'岭回归:  R²={r2_score(y_test, y_pred_ridge):.2f}',
    f'Lasso:  R²={r2_score(y_test, y_pred_lasso):.2f}',
    f'弹性网络: R²={r2_score(y_test, y_pred_elastic):.2f}'
])
plt.text(0.95, 0.15, text_str, transform=plt.gca().transAxes,
         ha='right', va='bottom', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.show()