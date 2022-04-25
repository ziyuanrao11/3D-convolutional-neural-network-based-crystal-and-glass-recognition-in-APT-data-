# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:33:34 2021

@author: z.rao
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
import os

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
#%%read the data
with h5py.File("GeWudata.h5", "r") as hf:    

    # Split the data into training/test features/targets
    X_train = hf["X_train_16"][0:30400]
    X_train = np.round(X_train)
    targets_train = hf["y_train_16"][0:30400]
    targets_train = np.round(targets_train)
    targets_train = np.int64(targets_train).flatten()
    
    X_test = hf["X_test_16"][0:14900] 
    X_test = np.round(X_test)
    targets_test = hf["y_test_16"][0:14900]
    targets_test = np.round(targets_test)
    targets_test = np.int64(targets_test).flatten()
    

#%%change numpy to tensor    
train_x = torch.from_numpy(X_train).float()
train_y = torch.from_numpy(targets_train)


test_x = torch.from_numpy(X_test).float()
test_y = torch.from_numpy(targets_test)

#%%define the hyperparameter
batch_size = 100 #We pick beforehand a batch_size that we will use for the training
n_cubic=16


# Pytorch train and test sets
train = torch.utils.data.TensorDataset(train_x,train_y)
test = torch.utils.data.TensorDataset(test_x,test_y)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

#define the convolutional autoecoder
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(2 * 2 * 2* 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
    
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 2 * 2 * 2 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 2, 2, 2))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, 
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1,
            output_padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(True),
            nn.ConvTranspose3d(8, 1, 3, stride=2, padding=1,
            output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        print(x.shape)
        x = self.unflatten(x)
        print(x.shape)
        x = self.decoder_conv(x)
        print(x.shape)
        x = torch.sigmoid(x)
        return x

#initial the loss function and optimizer
### Define the loss function
loss_fn = torch.nn.MSELoss()

### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
d = 4

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=d,fc2_input_dim=128)
decoder = Decoder(encoded_space_dim=d,fc2_input_dim=128)
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)

#train and evaluate the model
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer,noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        train = image_batch.view(batch_size,1,n_cubic,n_cubic,n_cubic)
        train = train.to(device)    
        # Encode data
        encoded_data = encoder(train)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, train)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

#test the data

### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.view(batch_size,1,n_cubic,n_cubic,n_cubic)
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

#see the reconstruction image


#train the data
num_epochs = 10
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss =train_epoch(encoder,decoder,device,
   train_loader,loss_fn,optim)
   val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['val_loss'].append(val_loss)

#plot the training history
test_epoch(encoder,decoder,device,test_loader,loss_fn).item()

# Plot losses
plt.figure(figsize=(10,8))
plt.semilogy(diz_loss['train_loss'], label='Train')
plt.semilogy(diz_loss['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
#plt.grid()
plt.legend()
#plt.title('loss')
plt.show()
plt.savefig('Figures/autoencoder_loss.png', format='png', dpi=300)
#%%plot t-sne figure
encoded_samples = []
for samples,labels in tqdm(test_loader):
    #print(sample[0])
    for i in range(len(samples)):
        img = samples[i].unsqueeze(0).to(device)
        img = img.view(1,1,n_cubic,n_cubic,n_cubic)
        label=labels[i].cpu().numpy()
        # Encode image
        encoder.eval()
        with torch.no_grad():
            encoded_img  = encoder(img)
        # Append to list
        encoded_img = encoded_img.flatten().cpu().numpy()
        encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
        encoded_sample['label'] = label
        encoded_samples.append(encoded_sample)
encoded_samples = pd.DataFrame(encoded_samples)
encoded_samples.to_csv('Figures/Minist_t-SNE.csv')
encoded_samples

#import plotly.express as px
plt.figure()
plt.scatter(x=encoded_samples['Enc. Variable 0'], y=encoded_samples['Enc. Variable 1'], 
           c=encoded_samples['label'], alpha=0.7, cmap='jet')
plt.show()
plt.savefig('Figures/Minist_t-SNE.png', format='png', dpi=300)

from sklearn.manifold import TSNE
plt.figure()
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
plt.scatter(x=tsne_results[:,0], y=tsne_results[:,1],
                 c=encoded_samples['label'], alpha=0.7, cmap='jet')
plt.show()
plt.savefig('Figures/Minist_t-SNE_1.png', format='png', dpi=300)
