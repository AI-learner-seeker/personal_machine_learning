# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标标签
feature_names = iris.feature_names  # 特征名称
target_names = iris.target_names  # 类别名称

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树分类器（设置最大深度为3防止过拟合）
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测测试集
y_pred = clf.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.2f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 可视化决策树
plt.figure(figsize=(15, 10))
plot_tree(clf, 
          filled=True, 
          feature_names=feature_names, 
          class_names=target_names,
          rounded=True)
plt.title("鸢尾花分类决策树")
plt.show()

# 输出特征重要性
print("\n特征重要性：")
for name, importance in zip(feature_names, clf.feature_importances_):
    print(f"{name}: {importance:.4f}")

