# KNN
```python 
class sklearn.neighbors.KNeighborsClassifier(
    n_neighbors=5,         # 邻居数，是KNN中最重要的参数
    weights='uniform', 
    algorithm='auto',      # 计算最近邻的算法，常用算法有{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    leaf_size=30, 
    p=2, 
    metric='minkowski', 
    metric_params=None, 
    n_jobs=None, 
    **kwargs
)
```
KNN算法的思想：
在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的前K个数据，
则该测试数据对应的类别就是K个数据中出现次数最多的那个分类。

其算法的描述为：
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。