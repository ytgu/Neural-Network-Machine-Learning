#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:48:58 2018

@author: oliviajin
"""

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

# process single subject
from val import preprocess

# process aggregate datasets
# from val_agg import preprocess

# random process
# from val_random import preprocess


# hyperparamters
batch_size = 100
learning_rate = 1e-4
num_epoches = 40
epohe_print = 2

# Update 03/13 split train and test in ensemble_process
file_name = 'project_datasets3/A04T_slice.mat'
# file_name = 'All datasets'
X_train, train_targets, X_val, val_targets, X_test, test_targets = preprocess(file_name, 17)
# X_train, train_targets, X_val, val_targets, X_test, test_targets = preprocess(7)

split = val_targets.shape[0]
test = test_targets.shape[0]

train_dataset = TensorDataset(X_train, train_targets)
val_dataset = TensorDataset(X_val, val_targets)
test_dataset = TensorDataset(X_test, test_targets)  # work around

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_loss = []
val_loss = []


# 定义 Recurrent Network 模型
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(in_dim, hidden_dim, n_layer, batch_first=True,
                            dropout=0.5, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
        #   self.hidden_dim)).cuda()
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out
    
def weights_init(m):
    if isinstance(m, nn.RNN):
        print('Initialize Weight')
        nn.init.orthogonal(m.weight_ih_l0.data)
        nn.init.orthogonal(m.weight_hh_l0.data)


model = Rnn(10, 128, 1, 4)
model.apply(weights_init)
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()
    print('Using GPU')

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-2)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay = 2e-2, nesterov = True)
# optimizer = optim.Adagrad(model.parameters(), lr=learning_rate, lr_decay=2e-2, weight_decay=2e-2)
count = 0
# 开始训练
for epoch in range(num_epoches):
    if epoch % epohe_print == 0:
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        feature, label = data
        # b, h, w = feature.size()
        if use_gpu:
            feature = Variable(feature).cuda()
            label = Variable(label).cuda()
        else:
            feature = Variable(feature)
            label = Variable(label)
        # 向前传播
        out = model(feature)
        loss = criterion(out, label)
        running_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / (batch_size * i)))
    if epoch % epohe_print == 0:
        print('Finish {} epoch\nTrain Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)),
            running_acc / (len(train_dataset))))
    train_loss.append(running_loss / (len(train_dataset)))

    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for data in val_loader:
        feature, label = data
        # b, h, w = feature.size()
        if use_gpu:
            feature = Variable(feature, volatile=True).cuda()
            label = Variable(label, volatile=True).cuda()
        else:
            feature = Variable(feature, volatile=True)
            label = Variable(label, volatile=True)
        out = model(feature)
        loss = criterion(out, label)
        eval_loss += loss.data[0] * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        # print('pred',pred,'out',out,'label',label)
        eval_acc += num_correct.data[0]
    if epoch % epohe_print == 0:
        print('Validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (split), eval_acc / (split)))
        print()
    val_loss.append(eval_loss / (split))
    if epoch > 1:
        if val_loss[epoch] > val_loss[epoch-1]:
            count +=1
    if count >= 100:
        break
    
test_loss = 0.
test_acc = 0.
for data in test_loader:
    feature, label = data
    # b, h, w = feature.size()
    if use_gpu:
        feature = Variable(feature, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        feature = Variable(feature, volatile=True)
        label = Variable(label, volatile=True)
    out = model(feature)
    loss = criterion(out, label)
    test_loss += loss.data[0] * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    # print('pred',pred,'out',out,'label',label)
    test_acc += num_correct.data[0]

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(test_loss / (test), test_acc / (test)))
print()

plt.figure()
plt.plot(train_loss, label = 'Train')
plt.plot(val_loss, label = 'Val')
plt.xlabel('Epoch')
plt.ylabel('Train/Validation Loss')
plt.legend()
plt.title(file_name)

plt.figure()
plt.plot(val_loss)
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title(file_name)

# 保存模型
# torch.save(model.state_dict(), './rnn.pth')
