import torch
import torch.nn as nn

class PredictionLSTM(nn.Module):
    def __init__(self,input_len, hidden_size, num_layers, dropout = 0.5):
        super(PredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_len,hidden_size,num_layers,dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden, cell):
        
        x = x.unsqueeze(1)
        #run data through LSTM
        out, (hidden, cell) = self.lstm(x.to(torch.float32),(hidden, cell))
        #get prediction from output of last timestep
        prediction  = self.fc(out[:,-1,:])
        #print(f'x shape {predc.shape}')
        return prediction, hidden, cell
    
class PairTradingLSTM(nn.Module):
    # initalize variables
    def __init__(self,input_len, hidden_size, num_layers, device, dropout = 0.5):
        super(PairTradingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # lstm model
        self.pred_lstm = PredictionLSTM(input_len,hidden_size,num_layers,dropout=dropout,)
        self.lstm = nn.LSTM(input_len,hidden_size,num_layers,dropout=dropout,
                            batch_first=True)
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
        
        out, (hidden, cell) = self.lstm(x.to(torch.float32), (hidden_states, cell_states))
        # extracting the hidden states the last timestep
        out = self.fc(out[:,-1,:])
        outputs.append(out)
        
        #predicts, n-1 = future, n-1 time step ahead since already predicted one above
        for i in range(future-1):
            
            out, hidden, cell = self.pred_lstm(out, hidden, cell)
            outputs.append(out)
            
        outputs = torch.cat(outputs, dim = 1) 
        
        return outputs
        
        