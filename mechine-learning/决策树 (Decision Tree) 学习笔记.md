# 决策树 (Decision Tree) 学习笔记

## 1. 决策树原理

### 1.1 什么是决策树

决策树是一种树形结构的预测模型，它通过一系列规则对数据进行分类或回归。树中的每个内部节点代表一个属性上的判断，每个分支代表一个判断结果的输出，每个叶节点代表一种分类结果。

决策树的优势：

- 易于理解和解释，可视化效果好
- 不需要预处理数据，对缺失值不敏感
- 可以处理非线性关系的数据
- 训练速度快，预测过程简单高效

### 1.2 决策树的构建过程

1. **特征选择**：选择最优特征作为当前节点的分裂标准
2. **决策树生成**：递归地生成子节点，直到满足停止条件
3. **决策树剪枝**：防止过拟合，提高泛化能力

### 1.3 特征选择的指标

#### 1.3.1 信息熵 (Information Entropy)

信息熵是度量样本集合纯度的指标，熵值越小，纯度越高。

对于一个包含k类样本的集合D，其信息熵定义为：

\(H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k\)

其中\(p_k\)是第k类样本在集合D中所占的比例。

#### 1.3.2 信息增益 (Information Gain)

信息增益是指特征A对训练数据集D的信息增益\(g(D,A)\)，定义为集合D的信息熵\(H(D)\)与特征A给定条件下D的条件熵\(H(D|A)\)之差：

\(g(D,A) = H(D) - H(D|A)\)

信息增益越大，表示使用特征A进行划分所获得的 "纯度提升" 越大。ID3 算法就是以信息增益作为特征选择的标准。

#### 1.3.3 增益率 (Gain Ratio)

C4.5 算法使用增益率来选择特征，以减少信息增益对可取值较多的特征的偏好。

\(g_R(D,A) = \frac{g(D,A)}{H_A(D)}\)

其中\(H_A(D) = -\sum_{i=1}^{n} \frac{|D_i|}{|D|} \log_2 \frac{|D_i|}{|D|}\)，称为特征A的固有值。

#### 1.3.4 基尼指数 (Gini Index)

CART 算法使用基尼指数来选择划分特征，基尼指数反映了从数据集D中随机抽取两个样本，其类别标记不一致的概率。

\(Gini(D) = 1 - \sum_{k=1}^{K} p_k^2\)

特征A的基尼指数定义为：

\(Gini\_index(D,A) = \sum_{i=1}^{n} \frac{|D_i|}{|D|} Gini(D_i)\)

### 1.4 决策树的剪枝

剪枝是为了防止决策树过拟合，提高泛化能力，主要分为：

- **预剪枝**：在决策树生成过程中，提前停止树的生长
- **后剪枝**：先生成完整的决策树，再对其进行剪枝

## 2. 决策树算法分类

| 算法 | 划分标准 | 树结构 | 适用问题   |
| ---- | -------- | ------ | ---------- |
| ID3  | 信息增益 | 多叉树 | 分类       |
| C4.5 | 增益率   | 多叉树 | 分类       |
| CART | 基尼指数 | 二叉树 | 分类、回归 |

## 3. 决策树代码实现

### 3.1 手动实现简单的 ID3 决策树

```python
import numpy as np
import pandas as pd
from collections import Counter

class ID3DecisionTree:
    def __init__(self):
        self.tree = None  # 存储决策树
    
    def calculate_entropy(self, y):
        """计算信息熵"""
        counter = Counter(y)
        entropy = 0.0
        for count in counter.values():
            p = count / len(y)
            entropy -= p * np.log2(p) if p > 0 else 0
        return entropy
    
    def calculate_information_gain(self, X, y, feature_index):
        """计算指定特征的信息增益"""
        base_entropy = self.calculate_entropy(y)
        feature_values = X[:, feature_index]
        unique_values = np.unique(feature_values)
        
        conditional_entropy = 0.0
        for value in unique_values:
            mask = (feature_values == value)
            subset_y = y[mask]
            p = len(subset_y) / len(y)
            conditional_entropy += p * self.calculate_entropy(subset_y)
        
        return base_entropy - conditional_entropy
    
    def choose_best_feature(self, X, y):
        """选择信息增益最大的特征作为划分依据"""
        num_features = X.shape[1]
        best_gain = -1
        best_feature_index = -1
        
        for i in range(num_features):
            gain = self.calculate_information_gain(X, y, i)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = i
        
        return best_feature_index
    
    def majority_vote(self, y):
        """多数表决确定叶节点类别"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def build_tree(self, X, y, feature_names):
        """递归构建决策树"""
        # 终止条件1：所有样本属于同一类别
        if len(np.unique(y)) == 1:
            return y[0]
        
        # 终止条件2：没有特征可划分
        if X.shape[1] == 0:
            return self.majority_vote(y)
        
        # 选择最佳划分特征
        best_feature_index = self.choose_best_feature(X, y)
        best_feature_name = feature_names[best_feature_index]
        
        # 构建树节点
        tree = {best_feature_name: {}}
        remaining_feature_names = [name for i, name in enumerate(feature_names) 
                                 if i != best_feature_index]
        
        # 划分数据集并递归构建子树
        feature_values = X[:, best_feature_index]
        unique_values = np.unique(feature_values)
        
        for value in unique_values:
            mask = (feature_values == value)
            X_subset = X[mask]
            y_subset = y[mask]
            X_subset = np.delete(X_subset, best_feature_index, axis=1)
            tree[best_feature_name][value] = self.build_tree(X_subset, y_subset, remaining_feature_names)
        
        return tree
    
    def fit(self, X, y, feature_names):
        """训练模型"""
        self.tree = self.build_tree(X, y, feature_names)
    
    def predict_sample(self, sample, tree, feature_names):
        """预测单个样本"""
        if not isinstance(tree, dict):
            return tree
        
        feature_name = next(iter(tree.keys()))
        feature_index = feature_names.index(feature_name)
        feature_value = sample[feature_index]
        
        if feature_value in tree[feature_name]:
            return self.predict_sample(sample, tree[feature_name][feature_value], feature_names)
        else:
            return self.majority_vote(list(tree[feature_name].values()))
    
    def predict(self, X, feature_names):
        """预测多个样本"""
        if self.tree is None:
            raise Exception("模型尚未训练，请先调用fit方法")
        
        return np.array([self.predict_sample(sample, self.tree, feature_names) for sample in X])

# 示例用法
if __name__ == "__main__":
    # 高尔夫数据集
    data = {
        '天气': ['晴朗', '晴朗', '多云', '下雨', '下雨', '下雨', '多云', '晴朗', '晴朗', '下雨', '晴朗', '多云', '多云', '下雨'],
        '温度': ['炎热', '炎热', '炎热', '温和', '凉爽', '凉爽', '凉爽', '温和', '凉爽', '温和', '温和', '温和', '炎热', '温和'],
        '湿度': ['高', '高', '高', '高', '正常', '正常', '正常', '高', '正常', '正常', '正常', '高', '正常', '高'],
        '风速': ['微风', '强风', '微风', '微风', '微风', '强风', '强风', '微风', '微风', '微风', '强风', '强风', '微风', '强风'],
        '是否打球': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '是', '是', '是', '否']
    }
    
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1].tolist()
    
    # 训练模型
    model = ID3DecisionTree()
    model.fit(X, y, feature_names)
    
    # 打印决策树
    print("决策树结构:")
    import pprint
    pprint.pprint(model.tree)
    
    # 预测新样本
    sample = ['晴朗', '温和', '正常', '强风']
    prediction = model.predict([sample], feature_names)
    print(f"\n样本 {sample} 的预测结果: {prediction[0]}")
    
```

### 3.2 使用 scikit-learn 实现决策树

```python
# 决策树分类示例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 创建并训练决策树分类器
# criterion: 划分标准，'gini'表示基尼指数，'entropy'表示信息熵
# max_depth: 树的最大深度，控制复杂度防止过拟合
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
print("决策树分类结果评估:")
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 输出决策树规则
tree_rules = export_text(clf, feature_names=feature_names)
print("\n决策树规则:")
print(tree_rules)

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.title("决策树分类模型")
plt.show()


# 决策树回归示例
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 加载数据集
housing = fetch_california_housing()
X_reg, y_reg = housing.data, housing.target
feature_names_reg = housing.feature_names

# 划分训练集和测试集
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 创建并训练决策树回归器
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
reg.fit(X_train_reg, y_train_reg)

# 预测
y_pred_reg = reg.predict(X_test_reg)

# 评估模型
print("\n决策树回归结果评估:")
print(f"均方误差(MSE): {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
print(f"R²得分: {r2_score(y_test_reg, y_pred_reg):.4f}")

# 可视化回归决策树
plt.figure(figsize=(15, 10))
plot_tree(reg, feature_names=feature_names_reg, filled=True)
plt.title("决策树回归模型")
plt.show()
```

## 4. 决策树的优缺点

### 4.1 优点

1. 易于理解和解释，可以可视化展示
2. 不需要预处理数据，对缺失值和异常值不敏感
3. 可以同时处理数值型和类别型特征
4. 训练速度快，预测过程简单高效
5. 可以捕捉特征之间的交互关系

### 4.2 缺点

1. 容易过拟合，泛化能力较差
2. 对噪声数据敏感
3. 可能会产生偏向于具有更多取值的特征的决策树
4. 是一种贪心算法，不能保证得到全局最优解
5. 类别不平衡时，可能会导致决策树有偏差

## 5. 决策树的应用场景

1. 信用风险评估
2. 客户分类与细分
3. 医疗诊断
4. 市场营销决策
5. 故障诊断
6. 推荐系统

## 6. 决策树的改进算法

1. **随机森林**：集成多个决策树，通过投票提高预测性能
2. **AdaBoost**：通过加权方式集成多个弱决策树
3. **GBDT**：梯度提升决策树，通过迭代方式改进模型
4. **XGBoost/LightGBM**：高效实现的梯度提升树算法，在各种竞赛中表现优异

这些集成算法通常能获得比单一决策树更好的性能，是实际应用中的首选。