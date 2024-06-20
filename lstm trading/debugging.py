import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision
import torchvision.models as models
from torchvision import *
from torchvision import datasets
from torchvision.models.feature_extraction import create_feature_extractor

from torchmetrics import *

from torch.utils.data import *

import sklearn as sk
from sklearn import *
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from itertools import compress


import matplotlib.pyplot as plt


import yfinance as yf

# time step to predict ahead, creates input sequences
# gonna have to figure out test
def create_lstm_data(data, time_step=1, future = 0,test_percent = .1):
    x_vec, y_vec = [], []
    # formats data so y = t and x = t-1, ... , t-time_steps
    for i in range(len(data) - time_step):
        split = i + time_step
        # checks if data is of same length for concatentation
        length = data[split : split + future, 0].shape[0]
        if(length == future):
            x_vec.append(data[i : split, 0].unsqueeze(0))
            y_vec.append(data[split : split + future, 0].unsqueeze(0))
    # calculate number of elements to allocate to test
    #dataset_length = len(x_vec)
    #num_of_train = dataset_length - (int)(test_percent * dataset_length)
    # concats x into matrix and y into vector, needs unsqueez to add single dimension for LSTM
    return torch.cat(x_vec,0).unsqueeze(-1), torch.cat(y_vec,0).unsqueeze(-1)#, torch.cat(x_vec[num_of_train:],0).unsqueeze(-1), torch.cat(y_vec[num_of_train:],0).unsqueeze(-1)

# create a dataset out of timeseries data, must be formatted first, tensor
# correct timeseries formatation
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,i):
        return self.X[i], self.y[i]

class PairTradingLSTM(nn.Module):
    # initalize variables
    def __init__(self,input_len, hidden_size, num_layers, device, dropout = 0.5):
        super(PairTradingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # lstm model
        self.lstm = nn.LSTM(input_len,hidden_size,num_layers,dropout=dropout,
                            batch_first = True)
        # outputs result
        self.fc = nn.Linear(hidden_size, 1)
    # forward pass
    def forward(self, x, future = 0):
        #to store predicted outputs
        outputs = []
        
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype = torch.float32).to(self.device)
        cell_states = torch.zeros(self.num_layers, x.size(0),self.hidden_size, dtype = torch.float32).to(self.device)
        # output is hidden state of the last layer
        # out = [batch_size, 1, hidden_size]
        # hidden = (hidden_cell, cell_states) for all layers of LSTM
        out, (hidden, cell) = self.lstm(x.to(torch.float32),(hidden_states, cell_states))
        # extracting the hidden states the last timestep
        out = self.fc(out[:,-1,:].squeeze(dim=1))
        outputs.append(out)
        
        #predicts, n = future, n time step ahead
        for i in range(future):
            out, (hidden, cell) = self.lstm(out,(hidden, cell))
            out  = self.linear(out[:,-1,:].squeeze(dim=1))
            
        outputs = torch.cat(outputs, dim = 1) 
        
        return outputs
def train_epoch(model,dataloader, future, loss_fn,optimizer):
    
    model.train()
    running_loss = 0.0

    for X, y in dataloader:
        X_train, y_train = X.to(device).to(torch.float32), y.to(device).to(torch.float32)
        
        output = model(X_train, future)
        print(f'{output.shape}')
        loss = loss_fn(output, y_train)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss = running_loss / len(train_dataloader)
    print(f'Train Average Loss: {avg_loss}')

# Download stock data from Yahoo Finance
hmc_data = yf.download('HMC', period='10y')['Close']
hymtf_data = yf.download('HYMTF', period='10y')['Close']
# get spread
spread = hmc_data - hymtf_data
spread = spread.values.reshape(-1,1)

#parameters for data
BATCH_SIZE = 64
time_step = 7
future = 14

scaler = MinMaxScaler(feature_range=(0,1))
#transforms/normalizes data and converts it to a tensor
normalized_data = torch.tensor(scaler.fit_transform(spread))
X_train, y_train = create_lstm_data(normalized_data, time_step, future)
#creates dataset
train_dataset = TimeSeriesDataset(X_train,y_train)
#test_dataset = TimeSeriesDataset(X_test,y_test)
#puts dataset into dataloader
train_dataloader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE)
#test_dataloader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE)
device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PairTradingLSTM(1,6,3,device).to(device)
lr = 0.001
num_epochs = 50
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)
for epoch in range(num_epochs):
    print(f'Epoch: {epoch}')
    train_epoch(model, train_dataloader, future, loss, optimizer)