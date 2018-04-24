import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from aggregate import preprocess
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from collections import Counter

# HYPERPARAMETERS VALUE
batch_size = 120
learning_rate = 1e-3
num_epoches = 10
epohe_print = 1
extend = 30
split = extend * 50
val_split = extend *50

X_train, X_val, X_test, train_targets, val_targets, test_targets, train_list, val_list, test_list = preprocess(extend)

train_dataset = TensorDataset(X_train, train_targets)
val_dataset = TensorDataset(X_val, val_targets) 
test_dataset = TensorDataset(X_test, test_targets)   

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)


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
  #      print(label_train[i, 0], result)
        if(result == label_new[0]):
            num_correct += 1
        else:
#             print(temp_train[i, :])
            pass
        count += index
      

    if epoch % epohe_print == 0:
        print('Epoch #{},\nTrain Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, training_loss / len(train_dataset),
            num_correct / len(train_list)
            )
        )
    
    
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
#        print(label_train[i, 0], result)
        if(result == label_new[0]):
            num_correct += 1
        else:
#            print(temp_train[i, :])
            pass
        count += index
        

    if epoch % epohe_print == 0:
        print('Validation  Loss: {:.6f}, Acc: {:.6f} \n'.format(val_loss / val_split, num_correct / len(val_list)))
    
    
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
#    print(label_train[i, 0], result)
    if(result == label_new[0]):
        num_correct += 1
    else:
#       print(temp_train[i, :])
        pass    
    count += index

    
print('Test  Loss: {:.6f}, Acc: {:.6f} \n'.format(test_loss / split, num_correct / 50))
