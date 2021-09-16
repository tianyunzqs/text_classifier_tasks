# -*- coding: utf-8 -*-
"""
 @Time    : 2018/11/16 09:58
 @Author  : hanzi5
 @Email   : hanzi5@yeah.net
 @File    : cart_classification.py
 @Software: PyCharm
"""
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


matplotlib.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算传入数据的Gini指数
def calcGini(x, y):
    unique_labels = set(y)
    sample_num = len(x)  # y总数据条数
    gini = 1.0
    for label in unique_labels:
        gini_k = len(x[y == label]) / sample_num  # y中每一个分类的概率（其实就是频率）
        gini -= gini_k ** 2
    return gini


# 计算Gini指数，选择最好的特征划分数据集，即返回最佳特征下标及传入数据集各列的Gini指数
def chooseBestFeature(dataSet, y):
    sample_num, feature_num = dataSet.shape
    columnFeaGini = {}  # 初始化参数，记录每一列x的每一种特征的基尼 Gini(D,A)
    for i in range(feature_num):  # 遍历所有x特征列
        featList = list(dataSet[:, i])  # 取这一列x中所有数据，转换为list类型
        prob = dict(Counter(featList))  # 使用Counter函数计算这一列x各特征数量
        for value in prob.keys():  # 循环这一列的特征，计算H(D|A)
            feaGini = 0.0
            bestFlag = 1.00001  # 对某一列x中，会出现x=是，y=是的特殊情况，这种情况下按“是”、“否”切分数据得到的Gini都一样，设置此参数将所有特征都乘以一个比1大一点点的值，但按某特征划分Gini为0时，设置为1
            # subDataSet1, sublen1 = splitDataSet(dataSet, i, value, 1)  # 获取切分后的数据
            # subDataSet2, sublen2 = splitDataSet(dataSet, i, value, 2)
            cls_same, cls_diff = dataSet[:, i] == value, dataSet[:, i] != value
            subDataSet1, sub_y1 = dataSet[cls_same], y[cls_same]
            subDataSet2, sub_y2 = dataSet[cls_diff], y[cls_diff]
            sublen1, sublen2 = len(subDataSet1), len(subDataSet2)
            if (sublen1 / sample_num) * calcGini(subDataSet1, sub_y1) == 0:  # 判断按此特征划分Gini值是否为0（全部为一类）
                bestFlag = 1
            feaGini += (sublen1 / sample_num) * calcGini(subDataSet1, sub_y1) + (sublen2 / sample_num) * calcGini(subDataSet2, sub_y2)
            columnFeaGini['%d_%s' % (i, value)] = feaGini * bestFlag
    bestFeature = min(columnFeaGini, key=columnFeaGini.get)  # 找到最小的Gini指数益对应的数据列
    return bestFeature, columnFeaGini


def createTree(dataSet, y, dim_list=None):
    """
    输入：训练数据集D，特征集A，阈值ε
    输出：决策树T
    """
    # y_lables = np.unique(dataSet[:, -1])
    y_lables = np.unique(y)

    # 1、如果数据集D中的所有实例都属于同一类label（Ck），则T为单节点树，并将类label（Ck）作为该结点的类标记，返回T
    if len(set(y_lables)) == 1:
        return y_lables[0]

    # 2、若特征集A=空，则T为单节点，并将数据集D中实例树最大的类label（Ck）作为该节点的类标记，返回T
    if len(dataSet[0]) == 0:
        labelCount = dict(Counter(y_lables))
        return max(labelCount, key=labelCount.get)

    # 3、否则，按CART算法就计算特征集A中各特征对数据集D的Gini，选择Gini指数最小的特征bestFeature（Ag）进行划分
    bestFeature, _ = chooseBestFeature(dataSet, y)

    dim_list = [i for i in range(len(dataSet[0]))] if not dim_list else dim_list
    # bestFeatureLable = features[int(bestFeature.split('_')[0])]  # 最佳特征
    best_feature_idx = dim_list[int(bestFeature.split('_')[0])]  # 最佳特征

    decisionTree = {best_feature_idx: {}}  # 构建树，以Gini指数最小的特征bestFeature为子节点
    # del (features[int(bestFeature.split('_')[0])])  # 该特征已最为子节点使用，则删除，以便接下来继续构建子树
    dim_list = dim_list[:best_feature_idx] + dim_list[best_feature_idx + 1:]

    # 使用beatFeature进行划分，划分产生2各节点，成树T，返回T
    y_lables_split = y[list(dataSet[:, int(bestFeature.split('_')[0])] == bestFeature.split('_')[1])]  # 获取按此划分后y数据列表
    y_lables_grp = dict(Counter(y_lables_split))  # 统计最优划分应该属于哪个i叶子节点“是”、“否”
    y_leaf = max(y_lables_grp, key=y_lables_grp.get)  # 获得划分后出现概率最大的y分类
    decisionTree[best_feature_idx][bestFeature.split('_')[1]] = y_leaf  # 设定左枝叶子值

    # 4、删除此最优划分数据x列，使用其他x列数据，递归地调用步1-3，得到子树Ti，返回Ti
    dataSetNew = np.delete(dataSet, int(bestFeature.split('_')[0]), axis=1)  # 删除此最优划分x列，使用剩余的x列进行数据划分
    # subFeatures = features[:]
    # 判断右枝类型，划分后的左右枝“是”、“否”是不一定的，所以这里进行判断
    y1 = y_lables[0]  # CART树y只能有2个分类
    y2 = y_lables[1]
    if y_leaf == y1:
        decisionTree[best_feature_idx][y2] = {}
        decisionTree[best_feature_idx][y2] = createTree(dataSetNew, y, dim_list)
    elif y_leaf == y2:
        decisionTree[best_feature_idx][y1] = {}
        decisionTree[best_feature_idx][y1] = createTree(dataSetNew, y, dim_list)

    return decisionTree


if __name__ == "__main__":
    X = [
        ["青年", "否", "否", "一般"],
        ["青年", "否", "否", "好"],
        ["青年", "是", "否", "好"],
        ["青年", "是", "是", "一般"],
        ["青年", "否", "否", "一般"],
        ["中年", "否", "否", "一般"],
        ["中年", "否", "否", "好"],
        ["中年", "是", "是", "好"],
        ["中年", "否", "是", "非常好"],
        ["中年", "否", "是", "非常好"],
        ["老年", "否", "是", "非常好"],
        ["老年", "否", "是", "好"],
        ["老年", "是", "否", "好"],
        ["老年", "是", "否", "非常好"],
        ["老年", "否", "否", "一般"]
    ]
    Y = ["否", "否", "是", "是", "否", "否", "否", "是", "是", "是", "是", "是", "是", "是", "否"]

    features = ['年龄', '工作', '房子', '信贷']
    dataSet = np.column_stack((np.array(X), np.array(Y)))

    # 结果验证，计算结果与《统计学习方法》李航，P71页一致。
    bestFeature, columnFeaGini = chooseBestFeature(np.array(X), np.array(Y))
    print('\nbestFeature:', bestFeature, '\nGini(D,A):', columnFeaGini)

    dt_Gini = createTree(np.array(X), np.array(Y))  # 建立决策树，CART分类树
    print('CART分类树：\n', dt_Gini)
