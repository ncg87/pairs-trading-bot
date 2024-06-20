import torch
import torch.nn as nn

# Encoder block of the model
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout = 0.5):
        super(Encoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout,
                           batch_first=True)

    def forward(self, x):
        
        outputs, (hidden, cell) = self.rnn(x)
        
        return hidden, cell
 
# Decoder block of the model   
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout = 0.5):
        super(Decoder,self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout,
                           )
        self.fc_out = nn.Linear(hidden_dim, 1)

    #predicts a single time step, with either y pred as input or y
    def forward(self, y, prev_hidden, prev_cell):

        y = y.unsqueeze(0)

        output, (hidden, cell) = self.rnn(y, (prev_hidden, prev_cell))

        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
    
class EncoderDecoderWrapper(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout = 0.5, output_size = 1, teacher_forcing=0.3, device = 'cpu'):
        super(EncoderDecoderWrapper,self).__init__()
        self.encoder = Encoder(input_dim,num_layers,hidden_dim,dropout)
        self.decoder = Decoder(input_dim,num_layers,hidden_dim,dropout)
        self.teacher_forcing = teacher_forcing
        self.device = device
        self.output_size = output_size

    def forward(self, source, target=None):
        
        # Number of elements in each batch
        batch_size = source.shape[0]
        
        # Output should be same size as target during training
        if(target != None):
            target_len = target.shape[1]
        
            assert(target_len == self.output_size)
        
        prev_hidden, prev_cell = self.encoder(source)

        prev_target = source[:,-1]
            
        outputs = torch.zeros(batch_size,self.output_size).to(self.device)
        
        for t in range(self.output_size):
            
            prediction, prev_hidden, prev_cell = self.decoder(prev_target, prev_hidden, prev_cell)

            outputs[:,t] = prediction.squeeze(1)
            
            # Chance of using actual next value vs chance of using predicted in training to predict next value 
            prev_target = target[:,t] if torch.rand(1) < self.teacher_forcing and target != None else prediction
        
        return outputs