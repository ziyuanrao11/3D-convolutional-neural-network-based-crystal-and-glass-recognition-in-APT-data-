# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 16:21:24 2021

@author: z.rao
"""

#!/usr/bin/python
# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn import metrics

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
import seaborn as sns

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import h5py
# from plot3D import *
#%%read the file
with h5py.File("GeWudata.h5", "r") as hf:    

    # Split the data into training/test features/targets
    #X_train = hf["X_train_20"][0:30400]
    #X_train = np.round(X_train)
    #targets_train = hf["y_train_20"][0:30400]
    #targets_train = np.round(targets_train)
    #targets_train = np.int64(targets_train).flatten()
    
    #X_test = hf["X_test_20"][0:14900] 
    #X_test = np.round(X_test)
    #targets_test = hf["y_test_20"][0:14900]
    #targets_test = np.round(targets_test)
    #targets_test = np.int64(targets_test).flatten()

    x_all_data = hf["x_all_8_data"]
    x_all_data = np.round(x_all_data)
    y_all_data = hf['y_all_8_data']
    y_all_data = np.round(y_all_data)
    y_all_data = np.int64(y_all_data).flatten()
    
bins     = np.arange(1000,25000,1000)
bin_y =  pd.DataFrame(y_all_data[:,])
y_binned = np.digitize(bin_y.index, bins, right=True)
#%%set the cross validation
for j in range(5):
    print("fold_var=", j)
    X_train, X_test, y_train, y_test = train_test_split(x_all_data, y_all_data, 
    test_size=0.33, random_state=j, stratify=y_binned)
    

    X_train=X_train[0:30400]
    y_train=y_train[0:30400]
    X_test=X_test[0:14900]
    y_test=y_test[0:14900]
    
    train_x = torch.from_numpy(X_train).float()
    train_y = torch.from_numpy(y_train)
    print(train_x.shape)


    test_x = torch.from_numpy(X_test).float()
    test_y = torch.from_numpy(y_test)
    print(test_x.shape)


    batch_size = 100 #We pick beforehand a batch_size that we will use for the training


    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(train_x,train_y)
    test = torch.utils.data.TensorDataset(test_x,test_y)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

    n_cubic=8
    conv=2

    # Create CNN Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            
            self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(True),
            #nn.Conv3d(16, 32, 3, stride=2, padding=0),
            #nn.BatchNorm3d(32),
            #nn.ReLU(True)
            )
            #self.conv_layer1 = self._conv_layer_set(1, 32)
            #self.conv_layer2 = self._conv_layer_set(32, 64)
            # self.conv_layer3 = self._conv_layer_set(64, 128)
            self.fc1 = nn.Linear(2**3*32, 128)
            self.fc2 = nn.Linear(128, 1)
            self.relu = nn.LeakyReLU()
            self.batch=nn.BatchNorm1d(128)
            self.drop=nn.Dropout(p=0.5)        
            
        #def _conv_layer_set(self, in_c, out_c):
            #conv_layer = nn.Sequential(
            #nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
            #nn.LeakyReLU(),
            #nn.MaxPool3d((2, 2, 2)),
            #)
            #return conv_layer
        

        def forward(self, x):
            # Set 1
            #out = self.conv_layer1(x)
            #out = self.conv_layer2(out)
            # out = self.conv_layer3(out)
            out = self.encoder_cnn(x)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.batch(out)
            out = self.drop(out)
            out = self.fc2(out)
            
            return out

    def change(predicted):
        for i in range(len(predicted)):
            if predicted[i] >= 0.5:
                predicted[i] = 1
            else:
                predicted[i] = 0
        return predicted
    #Definition of hyperparameters
    n_iters = 1520
    num_epochs = n_iters / (len(train_x) / batch_size)
    num_epochs = int(num_epochs)

    # Create CNN
    model = CNNModel()
    #model.cuda()
    print(model)

    # Cross Entropy Loss 
    # error = nn.CrossEntropyLoss()
    error = nn.BCEWithLogitsLoss()
    # error = nn.MSELoss()

    # SGD Optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # CNN model training
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            

            train = images.view(batch_size,1,n_cubic,n_cubic,n_cubic)
            # labels = Variable(labels)
            
            # labels = torch.flatten(labels)
            
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model(train)
            outputs = torch.flatten(outputs)
            #print(outputs)
            
            # Calculate softmax and ross entropy loss
            labels=labels.float()
            #print(labels)
            loss = error(outputs, labels)
            #print(loss)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()
            
            count += 1
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:
                    
                    test = images.view(batch_size,1,n_cubic,n_cubic,n_cubic)
                    # Forward propagation
                    outputs = model(test)
                    outputs = outputs.view(100)
                    m = nn.Sigmoid()
                    # Get predictions from the maximum value
                    predicted = m(outputs)
                    predicted = predicted.detach().numpy()
                    predicted = change(predicted)
                    # Total number of labels
                    total += len(labels)
                    correct += (predicted == labels.numpy()).sum()
                
                
                accuracy = 100 * correct / float(total)
                print(accuracy)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))
        train_x = train_x.view(30400,1,n_cubic,n_cubic,n_cubic)
        test_x = test_x.view(14900,1,n_cubic,n_cubic,n_cubic)
        train_y=train_y.float()
        test_y=test_y.float()
        train_ls.append(error(model(train_x).view(-1, 1), train_y.view(-1, 1)).item())
        print(train_ls)
        test_ls.append(error(model(test_x).view(-1, 1), test_y.view(-1, 1)).item())
        print(test_ls)
    #create ROC curve
    fpr, tpr, _ = metrics.roc_curve(test_y.view(-1, 1).numpy(),  model(test_x).view(-1, 1).detach().numpy())
    auc = metrics.roc_auc_score(test_y.view(-1, 1).numpy(),  model(test_x).view(-1, 1).detach().numpy())
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.plot(fpr,tpr, color="darkorange", lw=2, label="AUC="+str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("Receiver operating characteristic")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig('Figures/{}_{}_ROC_cv_{}.png'.format(n_cubic,conv,j), format='png', dpi=300)
    #plot loss
    sns.set(style='darkgrid')
    print ("plot curves")
    print(iteration_list)
    print(accuracy_list)
    print(loss_list)
    print(train_ls)
    print(test_ls)

    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1, num_epochs+1),train_ls, linewidth = 5)
    plt.plot(range(1, num_epochs+1),  test_ls, linewidth = 5, linestyle=':')
    plt.legend(['train loss', 'test loss'])
    #plt.text(1500, 0.8, 'Loss=%.4f' % test_ls[-1], fontdict={'size': 20, 'color':  'red'})
    if not os.path.isdir('Figures'):
        os.mkdir('Figures')
    plt.savefig('Figures/{}_{}_loss_cv_{}.png'.format(n_cubic,conv,j), format='png', dpi=300)
    #plot accuracy
    sns.set(style='darkgrid')
    print ("plot curves")
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(iteration_list,accuracy_list, linewidth = 5)
    plt.savefig('Figures/{}_{}_accuracy_cv_{}.png'.format(n_cubic,conv,j), format='png', dpi=300)
    #save mode
    torch.save(model.state_dict(), 'Figures/{}_{}_cv_{}.pt'.format(n_cubic,conv,j))
    d={'epoch': range(1, num_epochs+1), 'train_loss':train_ls, 'test_ls': test_ls}
    data=pd.DataFrame(data=d)
    data.to_csv('Figures/{}_{}_loss_cv_{}.csv'.format(n_cubic,conv,j))
    d={'epoch': iteration_list, 'accrracy_list':accuracy_list}
    data=pd.DataFrame(data=d)
    data.to_csv('Figures/{}_{}_accracy_cv_{}.csv'.format(n_cubic,conv,j))
    print ("=== train end ===")