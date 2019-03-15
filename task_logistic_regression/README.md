# 逻辑回归 (Logistic Regression)
```
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
