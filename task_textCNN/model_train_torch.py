# -*- coding: utf-8 -*-
# @Time        : 2022/6/22 14:10
# @Author      : tianyunzqs
# @Description :

import os
import json
import collections
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

categories = set()


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_json = json.loads(line.strip())
            data.append((line_json['sentence'], line_json['label']))
            categories.add(int(line_json['label']))
    return data


def build_vocab(data):
    vocab = collections.Counter()
    for sentence, label in data:
        vocab.update(sentence)
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    char2id = {'UNK': 0}
    char2id.update({char: i + 1 for i, (char, _) in enumerate(vocab)})
    id2char = {v: k for k, v in char2id.items()}
    return char2id, id2char


train_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'train.json'))
# import pandas as pd
# df = pd.DataFrame(train_data, columns=["text", "label"])
# df["len"] = df["text"].apply(len)
# print(df.describe())
valid_data = load_data(os.path.join(project_path, 'data', 'iflytek_public', 'dev.json'))
char2id, id2char = build_vocab(train_data)
vocab_size = len(char2id)
embed_size = 128
num_filter = 128
filter_size = [3, 4, 5]
MAX_LEN = 256


class TextCNNDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return len(train_data)

    @staticmethod
    def padding(sentence_input):
        if len(sentence_input) > MAX_LEN:
            return sentence_input[:MAX_LEN]
        else:
            return sentence_input + [0] * (MAX_LEN - len(sentence_input))

    def __getitem__(self, index):
        sentence, label = train_data[index]
        sentence_input = [char2id.get(char, 0) for char in sentence]
        sentence_input = torch.tensor(self.padding(sentence_input), dtype=torch.long)
        label_input = torch.tensor(int(label), dtype=torch.long)
        return sentence_input, label_input


train_dataset = TextCNNDataset()
train_dataloader = DataLoader(train_dataset, batch_size=16)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filter, (k, embed_size)) for k in filter_size])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filter * len(filter_size), len(categories))

    def conv_and_pool(self, x, conv):
        # print("x1", x, x.shape)
        x = F.relu(conv(x)).squeeze(3)
        # print("x", x.shape)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        # print("out1", out, out.shape)
        out = out.unsqueeze(1)
        # print("out2", out, out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


model = TextCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
print(f"Model structure: {model}\n\n")
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def evaluate():
    model.eval()
    true_labels, pred_labels = [], []
    for sentence, label in valid_data:
        sentence_input = [char2id.get(char, 0) for char in sentence]
        sentence_input = torch.tensor([sentence_input], dtype=torch.long)
        true_labels.append(int(label))
        pred = model(sentence_input)
        pred_labels.append(pred.detach().numpy().argmax())
    print(metrics.classification_report(np.array(true_labels), np.array(pred_labels)))
    val_acc = np.mean(np.equal(np.array(true_labels), np.array(pred_labels)))
    return val_acc


model.train()
for epoch in range(10):
    for batch, (X, y) in enumerate(train_dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            acc = np.mean(np.equal(np.argmax(pred.detach().numpy(), axis=1), y.detach().numpy()))
            print(f"epoch: {epoch} batch: {batch} loss: {loss.item()} acc: {acc}")
    print(evaluate())
