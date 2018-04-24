
# coding: utf-8



import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from Random_process import preprocess
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from collections import Counter

# HYPERPARAMETERS VALUE
batch_size = 120
learning_rate = 1e-3
num_epoches = 30
epohe_print = 1
extend = 30
split = extend * 50
val_split = extend * 50

path = 'project_datasets/'
filename = ['A01T_slice.mat','A02T_slice.mat', 'A03T_slice.mat', 'A04T_slice.mat', 'A05T_slice.mat',
           'A06T_slice.mat','A07T_slice.mat','A08T_slice.mat', 'A09T_slice.mat']

train_loss = []
train_acc = []
validation_loss = []
validation_acc = []
testing_loss = []
testing_acc = []
for files in filename:
    print('Running file: {}'.format(files))
    X_train, X_val, X_test, train_targets, val_targets, test_targets, train_list, val_list, test_list = preprocess(path+files, extend)

    train_dataset = TensorDataset(X_train, train_targets)
    val_dataset = TensorDataset(X_val, val_targets) 
    test_dataset = TensorDataset(X_test, test_targets)   

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    
    train_loss_file = []
    train_acc_file = []
    val_loss_file = []
    val_acc_file = []

    # MODEL DEFINITION
    class FCnet(nn.Module):
        def __init__(self, in_dim, n_class, dropout):
            super(FCnet, self).__init__()
            self.fc1 = nn.Linear(in_dim, 800)
            self.fc2 = nn.Linear(800, 100)
            self.fc3 = nn.Linear(100, n_class)
            self.dropout = nn.Dropout(dropout)
            self.batchnorm1 = nn.BatchNorm2d(800)
            self.batchnorm2 = nn.BatchNorm2d(100)

        def forward(self, x):  
            x = x.view(-1, self.num_flat_features(x))

            x = F.relu(self.batchnorm1(self.fc1(x)))
            x = self.dropout(x) 
            x = F.relu(self.batchnorm2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    

    # MODEL SETTING
    model = FCnet(300*10, 4, 0.5)
    cnn_flag = False

    loss_type = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-2)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9, 
    #                     weight_decay = 2e-2, nesterov = True)

    for epoch in range(num_epoches):

        # TRAINING STAGE
        model.train()
        temp = []
        training_loss = 0.0
        num_correct = 0.0
        for i, (feature, label) in enumerate(train_loader):
            if(cnn_flag == True):
                feature = feature.unsqueeze(1)
            feature, label = Variable(feature), Variable(label)

            # forward propagation
            output = model(feature)
            loss = loss_type(output, label)
            training_loss += loss.data[0] * label.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            temp.extend(pred.numpy())

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        label_old = train_targets.numpy()
        count = 0
        label_numpy = []
        temp_numpy = []
        temp = np.array(temp)
        for index in train_list:
            label_new = []
            label_new = label_old[count:count+index]
            temp_new = []
            temp_new = temp[count:count+index]
            result = np.argmax(np.bincount(temp_new.reshape(1,-1)[0]))
            if(result == label_new[0]):
                num_correct += 1
            else:
                pass
            count += index


        if epoch % epohe_print == 0:
            print('Epoch #{},\nTrain Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, training_loss / len(train_dataset),
                num_correct / len(train_list)                                 
                )
            )
            train_loss_file.append(training_loss / len(train_dataset))
            train_acc_file.append(num_correct / len(train_list))


        # VALIDATION STAGE
        model.eval()
        val_loss = 0.0
        num_correct = 0.0
        temp = []
        for (feature, label) in val_loader:
            if(cnn_flag == True):
                feature = feature.unsqueeze(1)
            feature, label = Variable(feature, volatile=True), Variable(label, volatile=True)

            # forward propagation
            output = model(feature)
            loss = loss_type(output, label)
            val_loss += loss.data[0] * label.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            temp.extend(pred.numpy())

        label_old = val_targets.numpy()
        count = 0
        label_numpy = []
        temp_numpy = []
        temp = np.array(temp)
        for index in val_list:
            label_new = []
            label_new = label_old[count:count+index]
            temp_new = []
            temp_new = temp[count:count+index]
            result = np.argmax(np.bincount(temp_new.reshape(1,-1)[0]))
            if(result == label_new[0]):
                num_correct += 1
            else:
                pass
            count += index


        if epoch % epohe_print == 0:
            print('Validation  Loss: {:.6f}, Acc: {:.6f} \n'.format(val_loss / val_split, num_correct / len(val_list)))
            val_loss_file.append(val_loss / val_split)
            val_acc_file.append(num_correct / len(val_list))

    # TEST STAGE
    model.eval()
    test_loss = 0.0
    num_correct = 0.0
    temp = []
    #count = 0
    for (feature, label) in test_loader:
        if(cnn_flag == True):
            feature = feature.unsqueeze(1)
        feature, label = Variable(feature, volatile=True), Variable(label, volatile=True)

        # forward propagation
        output = model(feature)
        loss = loss_type(output, label)
        test_loss += loss.data[0] * label.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        temp.extend(pred.numpy())

    label_old = test_targets.numpy()
    count = 0
    label_numpy = []
    temp_numpy = []
    temp = np.array(temp)
    for index in test_list:
        label_new = []
        label_new = label_old[count:count+index]
        temp_new = []
        temp_new = temp[count:count+index]
        result = np.argmax(np.bincount(temp_new.reshape(1,-1)[0]))
        if(result == label_new[0]):
            num_correct += 1
        else:
            pass    
        count += index


    print('Test  Loss: {:.6f}, Acc: {:.6f} \n'.format(test_loss / split, num_correct / 50))
    
    testing_loss.append(test_loss/split) 
    testing_acc.append(num_correct/50)
    train_loss.append(train_loss_file)
    train_acc.append(train_acc_file)
    validation_loss.append(val_loss_file)
    validation_acc.append(val_acc_file)





import colorsys
import matplotlib.pyplot as plt

N = 9
index = range(N)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
labels = ['A01T', 'A02T','A03T','A04T','A05T','A06T','A07T','A08T','A09T']
x_axis = range(1,31)
fig1 = plt.figure()
for i in range(N):
    plt.plot(x_axis, train_loss[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Training Loss')
plt.ylim(0.8,1.5)
plt.title('Training Loss of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper right")
plt.show()
fig1.savefig('Training_Loss_FC.png')

fig2 = plt.figure()
for i in range(N):
    plt.plot(x_axis, train_acc[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Training Acccuracy')
plt.ylim(0.2,1)
plt.title('Training Accuracy of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper left")
plt.show()
fig2.savefig('Training_Accuracy_FC.png')

fig3 = plt.figure()
for i in range(N):
    plt.plot(x_axis, validation_loss[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Validation Loss')
plt.ylim(1.3,1.5)
plt.xlim(1,30)
plt.title('Validation Loss of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper right")
plt.show()
fig3.savefig('Validation_Loss_FC.png')

fig4 = plt.figure()
for i in range(N):
    plt.plot(x_axis, validation_acc[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Validation Accuracy')
plt.ylim(0,1)
plt.xlim(0,30)
plt.title('Validation Accuracy of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper left")
plt.show()
fig4.savefig('Validation_Accuracy_FC.png')

print(testing_loss)
print(testing_acc)

