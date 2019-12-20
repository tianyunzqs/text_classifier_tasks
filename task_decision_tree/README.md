# 决策树（Decision Tree）
```python 
class sklearn.tree.DecisionTreeClassifier(
    criterion=’gini’,               # 该函数用于衡量分割的依据。常见的有"gini"用来计算基尼系数和"entropy"用来计算信息增益
    splitter=’best’, 
    max_depth=None,                 # 树的最大深度
    min_samples_split=2,            # 分割内部节点所需的最小样本数
    min_samples_leaf=1,             # 叶节点上所需的最小样本数
    min_weight_fraction_leaf=0.0, 
    max_features=None, 
    random_state=None, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    min_impurity_split=None, 
    class_weight=None, 
    presort=False
)
```

|算法|支持模型|树结构|特征选择|连续值处理|缺失值处理|剪枝|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
ID3|分类|多叉树|信息增益|不支持|不支持|不支持
C4.5|分类|多叉树|信息增益比|支持|支持|支持
CART|分类、回归|二叉树|基尼系数，均方差|支持|支持|支持

# 参考
1.[机器学习基础之一文读懂决策树](https://segmentfault.com/a/1190000020322548)<br/>
2.[机器学习之-常见决策树算法(ID3、C4.5、CART)](https://shuwoom.com/?p=1452)<br/>
3.[Python实现决策树2(CART分类树及CART回归树)](https://blog.csdn.net/weixin_43383558/article/details/84303339)<br>

