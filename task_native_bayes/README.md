# 高斯朴素贝叶斯 (GaussianNB)
高斯朴素贝叶斯的原理可以看这篇文章：http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
```python:
class sklearn.linear_model.LogisticRegression(
    penalty='l2',         # 惩罚项。一般都是"l1"或者"l2"。 
    dual=False,           # 这个参数仅适用于使用 liblinear 求解器的"l2"惩罚项。 一般当样本数大于特征数时，这个参数置为 False。
    tol=0.0001, 
    C=1.0,                # 正则化强度(较小的值表示更强的正则化)，必须是正的浮点数。
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight=None, 
    random_state=None, 
    solver='warn',        # 参数求解器。一般的有{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}。
    max_iter=100, 
    multi_class='warn',   # 多分类问题转化，如果使用 "ovr"，则是将多分类问题转换成多个二分类为题看待；如果使用 "multinomial"，损失函数则会是整个概率分布的多项式拟合损失。
    verbose=0, 
    warm_start=False, 
    n_jobs=None
)

```
# 多项式朴素贝叶斯 (MultinomialNB/MNB)
# 互补朴素贝叶斯 (ComplementNB/CMB)
ComplementNB 是标准多项式朴素贝叶斯(MNB)算法的一种改进，特别适用于不平衡数据集。
具体来说，ComplementNB 使用来自每个类的补充的统计信息来计算模型的权重。
CNB 的发明者通过实验结果表明，CNB 的参数估计比 MNB 的参数估计更稳定。
此外，在文本分类任务上，CNB 通常比 MNB 表现得更好(通常是相当大的优势)。
```python:
class sklearn.naive_bayes.ComplementNB(
    alpha=1.0,         # 加性(拉普拉斯/Lidstone)平滑参数(无平滑为0)
    fit_prior=True,    # 是否学习类先验概率。若为假，则使用统一先验
    class_prior=None,  # 类的先验概率。如果指定，则不根据数据调整先验
    norm=False         # 是否执行权重的第二次标准化
)
```
# 伯努利朴素贝叶斯 (BernoulliNB)
BernoulliNB 实现了基于多元伯努利分布的数据的朴素贝叶斯训练和分类算法。
BernoulliNB 可能在某些数据集上表现得更好，特别是那些文档较短的数据集。
BernoulliNB 的 sklearn 与上面介绍的算法接口相似。
