决策树(Decision Tree)是一种基于规则的基础而又经典的分类与回归方法，其模型结构呈现树形结构，可认为是一组if-then规则的集合。

决策树主要包含三个步骤：特征选择、决策树构建和决策树剪枝。

典型的决策树有ID3、C4.5和CART(Classification And Regression)，它们的主要区别在于树的结构与构造算法。其中ID3和C4.5只支持分类，而CART支持分类和回归。
# 1.决策树简介
## 1.1. ID3决策树
ID3决策树根据信息增益来构建决策树；
对训练集(或子集) *D*，计算其每个特征的信息增益，并比较大小，选择信息增益最大的特征作为该节点的特征，由该节点的不同取值建立子节点。再对子节点递归以上步骤，构建决策树；直到所有特征的信息增益均小于预设阈值或没有特征为止。

缺点： 信息增益偏向取值较多的特征
$g(D,A)=H(D)-H(D|A)=H(D)-\sum_{i=1}^{n}\frac{\left | D_{i} \right |}{\left | D \right |}H(D)=H(D)-(-\sum_{i=1}^{n}\frac{\left | D_{i} \right |}{\left | D \right |}\sum_{k=1}^{k}\frac{\left | D_{ik} \right |}{\left | D_{i} \right |}log_{2}\frac{\left | D_{ik} \right |}{\left | D_{i} \right |})$

举个例子，如果以身份证号作为一个属性去划分数据集，则数据集中有多少个人就会有多少个子类别，而每个子类别中只有一个样本，故$log_{2}\frac{\left | D_{ik} \right |}{\left | D_{i} \right |}=log_{2}\frac{1}{1}=0$ 则信息增益$g(D,A)=H(D)$，此时信息增益最大，选择该特征(身份证号)划分数据，然而这种划分毫无意义，但是从信息增益准则来讲，这就是最好的划分属性。

信息增益(information gain)表示得知特征 *X* 的信息而使得类 *Y*的信息的不确定性减少的程度。
## 1.2. C4.5决策树
C4.5决策树根据信息增益比来构建决策树；
C4.5算法与ID3算法只是将ID3中利用信息增益选择特征换成了利用信息增益比选择特征。
## CART决策树
CART决策树根据基尼系数来构建决策树。
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
CART|分类、回归|二叉树|基尼系数、均方差|支持|支持|支持

# 参考
1. [机器学习基础之一文读懂决策树](https://segmentfault.com/a/1190000020322548)<br/>
2. [机器学习之-常见决策树算法(ID3、C4.5、CART)](https://shuwoom.com/?p=1452)<br/>
3. [Python实现决策树2(CART分类树及CART回归树)](https://blog.csdn.net/weixin_43383558/article/details/84303339)<br>

