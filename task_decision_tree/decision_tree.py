# -*- coding: utf-8 -*-
# @Time        : 2019/12/18 14:36
# @Author      : tianyunzqs
# @Description :

import copy
from math import log
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'SimHei'  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


class DecisionTree(object):
    def __init__(self, decision_tree_type='CART', feature_list=None):
        self.decision_tree_type = decision_tree_type
        self.feature_list = feature_list

    @staticmethod
    def compute_entropy(x, y):
        """
        计算给定数据的信息熵 H(S) = -SUM(P*logP)
        :param x:
        :param y:
        :return:
        """
        sample_num = len(x)
        label_counter = Counter(y)
        dataset_entropy = 0.0
        for key in label_counter:
            prob = float(label_counter[key]) / sample_num
            dataset_entropy -= prob * log(prob, 2)  # get the log value

        return dataset_entropy

    def dataset_split_by_id3(self, x, y):
        """
        选择最好的数据集划分方式
        ID3算法：选择信息熵增益最大
        C4.5算法：选择信息熵增益比最大
        :param x:
        :param y:
        :return:
        """
        feature_num = len(x[0])
        base_entropy = self.compute_entropy(x, y)
        best_info_gain, best_info_gain_ratio = 0.0, 0.0
        best_feature_idx = -1
        for i in range(feature_num):
            unique_features = set([example[i] for example in x])
            new_entropy, split_entropy = 0.0, 0.0
            for feature in unique_features:
                sub_dataset, sub_labels = [], []
                for featVec, label in zip(x, y):
                    if featVec[i] == feature:
                        sub_dataset.append(list(featVec[:i]) + list(featVec[i + 1:]))
                        sub_labels.append(label)

                prob = len(sub_dataset) / float(len(x))
                new_entropy += prob * self.compute_entropy(sub_dataset, sub_labels)

            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = i

        return best_feature_idx

    def dataset_split_by_c45(self, x, y):
        """
        选择最好的数据集划分方式
        C4.5算法：选择信息熵增益比最大
        :param x:
        :param y:
        :return:
        """
        feature_num = len(x[0])
        base_entropy = self.compute_entropy(x, y)
        best_info_gain, best_info_gain_ratio = 0.0, 0.0
        best_feature_idx = -1
        for i in range(feature_num):
            unique_features = set([example[i] for example in x])
            new_entropy, split_entropy = 0.0, 0.0
            for feature in unique_features:
                sub_dataset, sub_labels = [], []
                for featVec, label in zip(x, y):
                    if featVec[i] == feature:
                        sub_dataset.append(list(featVec[:i]) + list(featVec[i + 1:]))
                        sub_labels.append(label)

                prob = len(sub_dataset) / float(len(x))
                new_entropy += prob * self.compute_entropy(sub_dataset, sub_labels)

                split_entropy += -prob * log(prob, 2)

            info_gain = base_entropy - new_entropy
            info_gain_ratio = info_gain / split_entropy if split_entropy else 0.0
            if info_gain_ratio > best_info_gain_ratio:
                best_info_gain_ratio = info_gain_ratio
                best_feature_idx = i

        return best_feature_idx

    def create_tree_by_id3_and_c45(self, x, y, feature_list=None):
        """
        创建决策树
        :param x:
        :param y:
        :param feature_list:
        :return:
        """
        # the type is the same, so stop classify
        if len(set(y)) <= 1:
            return y[0]
        # traversal all the features and choose the most frequent feature
        if len(x[0]) == 1:
            return Counter(y).most_common(1)

        feature_list = [i for i in range(len(y))] if not feature_list else feature_list
        if self.decision_tree_type == 'ID3':
            best_feature_idx = self.dataset_split_by_id3(x, y)
        elif self.decision_tree_type == 'C45':
            best_feature_idx = self.dataset_split_by_c45(x, y)
        else:
            raise KeyError
        best_feature = feature_list[int(best_feature_idx)]  # 最佳特征
        decision_tree = {best_feature: {}}
        # feature_list = feature_list[:best_feature_idx] + feature_list[best_feature_idx + 1:]
        feature_list.pop(int(best_feature_idx))
        # get the list which attain the whole properties
        best_feature_values = set([sample[best_feature_idx] for sample in x])
        for value in best_feature_values:
            sub_dataset, sub_labels = [], []
            for featVec, label in zip(x, y):
                if featVec[best_feature_idx] == value:
                    sub_dataset.append(list(featVec[:best_feature_idx]) + list(featVec[best_feature_idx + 1:]))
                    sub_labels.append(label)

            decision_tree[best_feature][value] = self.create_tree_by_id3_and_c45(sub_dataset, sub_labels, feature_list)

        return decision_tree

    @staticmethod
    def compute_gini(x, y):
        """
        计算数据集x的基尼指数
        :param x: 数据集
        :param y: 数据集对应的类别标签
        :return: 该数据集的gini指数
        """
        unique_labels = set(y)
        sample_num = len(x)  # y总数据条数
        gini = 1.0
        for label in unique_labels:
            gini_k = len(x[y == label]) / sample_num  # y中每一个分类的概率（其实就是频率）
            gini -= gini_k ** 2
        return gini

    def dataset_split_by_cart(self, x, y):
        """
        选择最好的特征划分数据集，即返回最佳特征下标
        :param x:
        :param y:
        :return:
        """
        sample_num, feature_num = x.shape
        column_feature_gini = {}  # 初始化参数，记录每一列x的每一种特征的基尼 Gini(D,A)
        for i in range(feature_num):  # 遍历所有x特征列
            column_i = dict(Counter(x[:, i]))  # 使用Counter函数计算这一列x各特征数量
            for value in column_i.keys():  # 循环这一列的特征，计算H(D|A)
                # 对某一列x中，会出现x=是，y=是的特殊情况，这种情况下按“是”、“否”切分数据得到的Gini都一样，
                # 设置此参数将所有特征都乘以一个比1大一点点的值，但按某特征划分Gini为0时，设置为1
                best_flag = 1.00001
                cls_same, cls_diff = x[:, i] == value, x[:, i] != value
                sub_x1, sub_y1 = x[cls_same], y[cls_same]
                sub_x2, sub_y2 = x[cls_diff], y[cls_diff]
                sublen1, sublen2 = len(sub_x1), len(sub_x2)
                # 判断按此特征划分Gini值是否为0（全部为一类）
                if (sublen1 / sample_num) * self.compute_gini(sub_x1, sub_y1) == 0:
                    best_flag = 1
                feaGini = (sublen1 / sample_num) * self.compute_gini(sub_x1, sub_y1) + \
                          (sublen2 / sample_num) * self.compute_gini(sub_x2, sub_y2)
                column_feature_gini[(i, value)] = feaGini * best_flag

        # 找到最小的Gini指数益对应的数据列
        best_feature_and_idx = min(column_feature_gini, key=column_feature_gini.get)

        return best_feature_and_idx, column_feature_gini

    def create_tree_by_cart(self, x, y, feature_list=None):
        """
        输入：训练数据集D，特征集A，阈值ε
        输出：决策树T
        """
        y_lables = np.unique(y)

        # 1、如果数据集D中的所有实例都属于同一类label（Ck），则T为单节点树，并将类label（Ck）作为该结点的类标记，返回T
        if len(set(y_lables)) == 1:
            return y_lables[0]

        # 2、若特征集A=空，则T为单节点，并将数据集D中实例树最大的类label（Ck）作为该节点的类标记，返回T
        if len(x[0]) == 0:
            labelCount = dict(Counter(y_lables))
            return max(labelCount, key=labelCount.get)

        # 3、否则，按CART算法就计算特征集A中各特征对数据集D的Gini，选择Gini指数最小的特征bestFeature（Ag）进行划分
        best_feature_and_idx, _ = self.dataset_split_by_cart(x, y)

        feature_list = [i for i in range(len(x[0]))] if not feature_list else feature_list
        best_feature_idx = feature_list[int(best_feature_and_idx[0])]  # 最佳特征

        decision_tree = {best_feature_idx: {}}  # 构建树，以Gini指数最小的特征bestFeature为子节点
        # feature_list = feature_list[:best_feature_idx] + feature_list[best_feature_idx + 1:]
        feature_list.pop(int(best_feature_and_idx[0]))

        # 使用beatFeature进行划分，划分产生2个节点，成树T，返回T
        y_lables_split = y[list(x[:, int(best_feature_and_idx[0])] == best_feature_and_idx[1])]  # 获取按此划分后y数据列表
        y_lables_grp = Counter(y_lables_split)  # 统计最优划分应该属于哪个i叶子节点“是”、“否”
        y_leaf = y_lables_grp.most_common(1)[0][0]  # 获得划分后出现概率最大的y分类
        decision_tree[best_feature_idx][best_feature_and_idx[1]] = y_leaf  # 设定左枝叶子值

        # 4、删除此最优划分数据x列，使用其他x列数据，递归地调用步1-3，得到子树Ti，返回Ti
        sub_x = np.delete(x, int(best_feature_and_idx[0]), axis=1)  # 删除此最优划分x列，使用剩余的x列进行数据划分
        # 判断右枝类型，划分后的左右枝“是”、“否”是不一定的，所以这里进行判断
        y1 = y_lables[0]  # CART树y只能有2个分类
        y2 = y_lables[1]
        if y_leaf == y1:
            decision_tree[best_feature_idx][y2] = self.create_tree_by_cart(sub_x, y, feature_list)
        elif y_leaf == y2:
            decision_tree[best_feature_idx][y1] = self.create_tree_by_cart(sub_x, y, feature_list)

        return decision_tree

    def train(self, x, y):
        x, y = np.array(x), np.array(y)
        if self.decision_tree_type in ('ID3', 'C45'):
            return self.create_tree_by_id3_and_c45(x, y, feature_list=copy.deepcopy(self.feature_list))
        elif self.decision_tree_type == 'CART':
            return self.create_tree_by_cart(x, y, feature_list=copy.deepcopy(self.feature_list))
        else:
            raise KeyError

    def predict(self, tree, x):
        """
        决策树预测
        :param tree:
        :param x:
        :return:
        """
        root = list(tree.keys())[0]
        root_dict = tree[root]
        for key, value in root_dict.items():
            if x[root if not self.feature_list else self.feature_list.index(root)] == key:
                if isinstance(value, dict):
                    _label = self.predict(value, x)
                else:
                    _label = value

                return _label

        raise KeyError

    def getNumLeafs(self, myTree):
        numLeafs = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if isinstance(secondDict[key], dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                numLeafs += self.getNumLeafs(secondDict[key])
            else:
                numLeafs += 1
        return numLeafs

    def getTreeDepth(self, myTree):
        maxDepth = 0
        firstStr = list(myTree.keys())[0]  # myTree.keys()[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if isinstance(secondDict[key], dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                thisDepth = 1 + self.getTreeDepth(secondDict[key])
            else:
                thisDepth = 1
            if thisDepth > maxDepth: maxDepth = thisDepth
        return maxDepth

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                                xytext=centerPt, textcoords='axes fraction',
                                va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

    def plotTree(self, myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
        numLeafs = self.getNumLeafs(myTree)  # this determines the x width of this tree
        # depth = getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]  # myTree.keys()[0]     #the text label for this node should be this
        cntrPt = (self.xOff + (1.0 + float(numLeafs)) / 2.0 / self.totalW, self.yOff)
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, decisionNode)
        secondDict = myTree[firstStr]
        self.yOff = self.yOff - 1.0 / self.totalD
        for key in secondDict.keys():
            if isinstance(secondDict[key], dict):  # test to see if the nodes are dictonaires, if not they are leaf nodes
                self.plotTree(secondDict[key], cntrPt, str(key))  # recursion
            else:  # it's a leaf node print the leaf node
                self.xOff = self.xOff + 1.0 / self.totalW
                self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, leafNode)
                self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
        self.yOff = self.yOff + 1.0 / self.totalD

    def createPlot(self, myTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
        # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
        self.totalW = float(self.getNumLeafs(myTree))
        self.totalD = float(self.getTreeDepth(myTree))
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0
        self.plotTree(myTree, (0.5, 1.0), '')
        plt.show()


if __name__ == '__main__':
    # dt = DecisionTree(decision_tree_type='ID3')
    #
    # X = [
    #     [1, 0, 1],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [0, 0, 1],
    #     [0, 1, 1]
    # ]
    # Y = [
    #     'yes',
    #     'yes',
    #     'no',
    #     'no',
    #     'no'
    # ]
    #
    # myTree = dt.train(X, Y)
    # print(myTree)
    # print(Y)
    # predict_label = dt.predict(myTree, [1, 1, 1])
    # print(predict_label)

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

    X_test = [
        ["青年", "否", "是", "一般"],
        ["中年", "是", "否", "好"],
        ["老年", "否", "是", "一般"],
    ]

    dt = DecisionTree(decision_tree_type='C45', feature_list=['年龄', '工作', '房子', '信贷'])
    myTree = dt.train(X, Y)
    print(myTree)
    # predict_label = dt.predict(myTree, X_test[1])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[2])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[1])
    # print(predict_label)
    # predict_label = dt.predict(myTree, X_test[2])
    # print(predict_label)
    dt.createPlot(myTree)
