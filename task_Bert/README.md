# Bert

Bert的核心过程：  
1）先从数据集中抽取两个句子，其中第二句是第一句的下一句的概率是50%，这样就能学习句子之间的关系；  
2）随机地遮掩/替换句子中一些词语，并要求模型对其进行预测，这样就能学习到句子内部的关系。  

Bert模型分为两个阶段：pre-training和fine-tuning
## pre-training
为了兼顾NLP四大类任务，bert模型在预训练阶段构造了如下两个训练任务：
### Mask LM
Mask LM是谷歌提出的一种学习句子内部关系的一种trick，该trick的灵感来源于完形填空（跟word2vec的CBOW模型类似，参考1.5节），基本思想是随机遮掩句子中一些词，并利用上下文信息对其进行预测。使用该trick后可以使得模型对上下文信息具有完全意义上的双向表征。

然而，该trick具有以下两个缺点：  
1）pre-training和fine-tuning阶段不一致，该trick在fine-tuning阶段是不可见的，只有在pre-training阶段才是用该trick。谷歌给出的解决方案：随机替换句子中15%的词语，其中80%时间用【MASK】替换该词，10%时间用随机词替换，10%时间保持不变；  

Q: 为什么要以一定的概率保持不变呢？   
A：如果100%的概率都用[MASK]来取代被选中的词，那么在fine tuning的时候模型可能会有一些没见过的词。那么为什么要以一定的概率使用随机词呢？这是因为Transformer要保持对每个输入token分布式的表征，否则Transformer很可能会记住这个[MASK]就是"hairy"。至于使用随机词带来的负面影响，文章中说了,所有其他的token(即非"hairy"的token)共享15%*10% = 1.5%的概率，其影响是可以忽略不计的。  

2） 模型收敛速度慢。在每个batch中只有15%的token被预测，因此需要在pre-training阶段花费更多的训练次数。  
### Next Sentence Prediction
在自动问答（QA）、自然语言理解（NLI）等任务中，需要弄清上下两句话之间的关系，为了是模型理解这种关系，故需要训练Next Sentence Prediction。构造训练语料方式：50%时间下一句是真正的下一句，50%时间下一句是语料中随机的一句话。  

在pre-training阶段，模型的损失函数是Mask LM和Next Sentence Prediction最大似然概率均值之和。
## fine-tuning
当pre-training训练完后（谷歌已提供训练好的模型供下载使用），在下游的NLP任务只需进行简单的fine-tuning即可训练模型（谷歌已提供样例）。

## Bert模型输入
Bert的输入部分是个线性序列，两个句子通过分隔符分割，最前面和最后增加两个标识符号。对于单句输入类型任务，如：序列标注和文本分类等，只需用到embeddingA  

每个单词有三个embedding：  
1）位置信息embedding（position embeddings），表示当前词所在位置的index embedding，这是因为NLP中单词顺序是很重要的特征，需要在这里对位置信息进行编码；  

2）句子embedding（segmentation embeddings ），表示当前词所在句子的index embedding，因为前面提到训练数据都是由两个句子构成的，那么每个句子有个句子整体的embedding项对应给每个单词。  

3）单词embedding（token embeddings），这个就是我们之前一直提到的单词embedding，表示当前词的embedding；  
把单词对应的三个embedding叠加，就形成了Bert的输入。