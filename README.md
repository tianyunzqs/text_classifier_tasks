# text_classifier_tasks
文本分类方法集合

## 1. 环境说明
tensorflow==1.14.0  
keras==2.3.1

## 2.TfidfVectorizer、CountVectorizer 和 TfidfTransformer
TfidfVectorizer、CountVectorizer 和 TfidfTransformer 是 sklearn 中处理自然语言常用的工具。  
TfidfVectorizer 相当于 CountVectorizer + TfidfTransformer

### 2.1. CountVectorizer
CountVectorizer 的作用是将文本文档转换为计数的稀疏矩阵。下面举一个具体的例子来说明(代码来自于官方文档)。
```python
from sklearn.feature_extraction.text import CountVectorizer
# 定义一个 list，其中每个元素是一个文档(一个句子)
corpus = [
  'This is the first document.',
  'This document is the second document.',
  'And this is the third one.',
  'Is this the first document?',
]
vectorizer = CountVectorizer()
# 将文本数据转换为计数的稀疏矩阵
X = vectorizer.fit_transform(corpus)
# 查看每个单词的位置
print(vectorizer.get_feature_names())
# 输出: ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']

# 由于 X 存储为稀疏矩阵，需要转换为 array 才能查看
print(X.toarray())
# 输出:
# [[0 1 1 1 0 0 1 0 1]
# [0 2 0 1 0 1 1 0 1]
# [1 0 0 1 1 0 1 1 1]
# [0 1 1 1 0 0 1 0 1]]
```

`vectorizer.get_feature_names()` 包含了数据中出现的所有单词去重后的集合，相当于一个词典。  
当然你也可以给 CountVectorizer 提供一个单独的词典，否则 CountVectorizer 会自己从数据中学习到词典。  

`X.toarray()` 是查看文档转化后的计数矩阵。  
比如矩阵的第一行 `[0 1 1 1 0 0 1 0 1]` 对应于文档中的第一句 "This is the first document."，表示词典中对应位置的单词出现的次数。  
`X.toarray()` 的维度是 (4,9)，可以看到转化之后的计数矩阵的元素是 4，每个元素的长度固定为 9，这里的 9 就是字典的长度。  

### 2.2. TfidfTransformer
使用 TfidfTransformer 如下，输出的 tf-idf 矩阵维度也是 (4,9)
```python
from sklearn.feature_extraction.text import TfidfTransformer

transform = TfidfTransformer()    
Y = transform.fit_transform(X)   # 这里的输入是上面文档的计数矩阵
print(Y.toarray())               # 输出转换为tf-idf后的 Y 矩阵
# 输出为：
# [[0.        0.46979139 0.58028582 0.38408524 0.         0.         0.38408524 0.         0.38408524]
# [0.         0.6876236  0.         0.28108867 0.         0.53864762 0.28108867 0.         0.28108867]
# [0.51184851 0.         0.         0.26710379 0.51184851 0.         0.26710379 0.51184851 0.26710379]
# [0.         0.46979139 0.58028582 0.38408524 0.         0.         0.38408524 0.         0.38408524]]
```

### 2.3. TfidfVectorizer
TfidfVectorizer 相当于 CountVectorizer 和 TfidfTransformer 的结合使用。  
上面代码先调用了 CountVectorizer，然后调用了 TfidfTransformer。使用 TfidfVectorizer 可以简化代码如下：  
```python
# 把每个设备的 app 列表转换为字符串，以空格分隔
apps=deviceid_packages['apps'].apply(lambda x:' '.join(x)).tolist()
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
# 原来的 app 列表 转换为计数的稀疏矩阵。
cntTf = vectorizer.fit_transform(apps)
# 得到 tf-idf 矩阵
tfidf=transformer.fit_transform(cntTf)
# 得到所有的 APP 列表，相当于词典
word=vectorizer.get_feature_names()
```

## 3. 模型结果

|模型|accuracy|备注|
| :---: | :---: | :---: | 
|knn|0.5514||
|decision tree|0.4317||
|random forest|0.4856||
|logistic regression|0.5133||
|native bayes|0.5383||
|svm|0.5583||
|xgboost|0.5190||
|lightgbm|0.5106||
|fasttext|0.5329||
|textcnn|0.5468||
|bert|0.6098||
|bert+textcnn|0.5937|将bert当word-embedding，<br>textcnn输入为bert输出|
|bert+textcnn2|0.5972|将bert输出cls拼接textcnn输出，<br>textcnn输入为bert输入中token经过独立embedding|
|bilstm|0.4821||
|bilstm+attention|0.5264||
|nezha|0.6110||
|bert+rdrop|0.6152||
|bert+adversarial_training|0.6110||
|bert+gradient_penalty|0.6137||
|GAU|0.6172||

说明1：以上结果未经过详尽的调参，只是一个参考结果，调参后可能效果更佳~  
说明2：bert+textcnn效果变差，可参考[https://www.zhihu.com/question/477075127](https://www.zhihu.com/question/477075127)  
