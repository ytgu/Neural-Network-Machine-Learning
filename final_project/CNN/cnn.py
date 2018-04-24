
# coding: utf-8

# CNN




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
learning_rate = 1e-4
num_epoches = 50
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
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight.data)
    #         nn.init.xavier_normal(m.bias.data)


    class Cnn(nn.Module):
        def __init__(self):
            super(Cnn, self).__init__()

            self.conv1 = nn.Conv2d(1, 24, (12,5))
            self.pool = nn.MaxPool2d((2,1))
            self.conv2 = nn.Conv2d(24, 48, (10,6))
            self.fc1 = nn.Linear(48*67*1, 500)  #two layer
            self.fc2 = nn.Linear(500, 4)
            self.dropout = nn.Dropout(0.5)
            self.batchnorm1 = nn.BatchNorm2d(24)
            self.batchnorm2 = nn.BatchNorm2d(48)

        def forward(self, x):
            x = self.conv1(x)
            x = self.batchnorm1(x)
            x = self.pool(x)
            x = F.relu(x)
            #print x.shape
            x = self.dropout(x)

            x = self.conv2(x)
            x = self.batchnorm2(x)
            x = self.pool(x)
            x = F.relu(x)

            #print x.shape
            x = x.view(-1, 48*67*1)  
            x = self.dropout(x)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)

            return x

    # MODEL SETTING
    model = Cnn()
    model.apply(weights_init)
    cnn_flag = True

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
            #num_correct += pred.eq(label.data.view_as(pred)).long().cpu().sum()
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
      #      print(label_new[0], result)
            if(result == label_new[0]):
                num_correct += 1
            else:
    #             print(temp_new)
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
    #        print(label_new[0], result)
            if(result == label_new[0]):
                num_correct += 1
            else:
    #            print(temp_new)
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
    #    print(label_new[0], result)
        if(result == label_new[0]):
            num_correct += 1
        else:
    #       print(temp_new)
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
x_axis = range(1,51)
fig1 = plt.figure()
for i in range(N):
    plt.plot(x_axis, train_loss[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Training Loss')
plt.ylim(1,1.5)
plt.title('Training Loss of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="lower left")
plt.show()
fig1.savefig(path+'Training Loss.png')

fig2 = plt.figure()
for i in range(N):
    plt.plot(x_axis, train_acc[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Training Acccuracy')
plt.ylim(0,1)
plt.title('Training Accuracy of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper left")
plt.show()
fig2.savefig(path+'Training Accuracy.png')

fig3 = plt.figure()
for i in range(N):
    plt.plot(x_axis, validation_loss[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Validation Loss')
plt.ylim(1,1.5)
plt.xlim(1,60)
plt.title('Validation Loss of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="upper right")
plt.show()
fig3.savefig(path+'Validation Loss.png')

fig4 = plt.figure()
for i in range(N):
    plt.plot(x_axis, validation_acc[i], color=RGB_tuples[i], label = labels[i])
plt.xlabel('Number of epoch')     
plt.ylabel('Validation Accuracy')
plt.ylim(0,1)
plt.xlim(0,65)
plt.title('Validation Accuracy of CNN for all subjects')
plt.grid(color='0.7', linestyle='-', linewidth=1)
plt.legend(loc="lower right")
plt.show()
fig4.savefig(path+'Validation Accuracy.png')

print(testing_loss)
print(testing_acc)

