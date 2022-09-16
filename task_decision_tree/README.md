[https://blog.csdn.net/tianyunzqs/article/details/103692986?spm=1001.2014.3001.5502](https://blog.csdn.net/tianyunzqs/article/details/103692986?spm=1001.2014.3001.5502)  
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
