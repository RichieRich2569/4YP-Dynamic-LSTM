"""LSTM + CNN Network Model"""

import torch.nn as nn
import torch
from constants import *
    
class Model(nn.Module):
    def __init__(self, device, input_dim, l1_out, lstm_out, l2_out, decoder_chan_init=1):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.l1_out = l1_out
        self.lstm_out = lstm_out
        self.l2_out = l2_out
        
        modules = []
        
        # First linear layer
        layer = nn.Linear(input_dim, l1_out[0])
        modules.append(layer)
        for i in range(1,len(l1_out)):
            layer = nn.Linear(l1_out[i-1],l1_out[i])
            modules.append(layer)
        modules.append(nn.ReLU())
            
        self.l1_net = nn.Sequential(*modules)
        modules_LSTM = []
        
        # LSTM layer
        layer = nn.LSTM(l1_out[-1], lstm_out[0], batch_first=True).to(self.device)
        modules_LSTM.append(layer)
        for i in range(1,len(lstm_out)):          
            layer = nn.LSTM(lstm_out[i-1],lstm_out[i], batch_first=True).to(self.device)
            modules_LSTM.append(layer)
        self.modules_LSTM = nn.ModuleList(modules_LSTM)
            
        # Second and third linear layer
        modules1 = [nn.ReLU()]
        modules2 = [nn.ReLU()]
        layer1 = nn.Linear(lstm_out[-1], l2_out[0])
        layer2 = nn.Linear(lstm_out[-1], l2_out[0])
        modules1.append(layer1)
        modules2.append(layer2)
        for i in range(1,len(l2_out)):
            layer1 = nn.Linear(l2_out[i-1],l2_out[i])
            layer2 = nn.Linear(l2_out[i-1],l2_out[i])
            modules1.append(layer1)
            modules2.append(layer2)
            
        self.l2_net = nn.Sequential(*modules1)
        self.l3_net = nn.Sequential(*modules2)
        
        # Variational auto-encoder
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.sampling_loss = 0
            
        # Decoder
        s = int((l2_out[-1]/decoder_chan_init)**(1/2))
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, torch.Size([decoder_chan_init,s,s])),
            nn.ConvTranspose2d(decoder_chan_init,8,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8,4,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4,1,kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
    def forward(self, x, hidden= None, return_hidden = False):
        # Define x with shape (N,T,D)
        s = self.l1_net(x)
        
        # Pass through LSTMs
        if hidden is None:
            hidden = self.init_states(N=s.shape[0])
        final_hidden_states = []
        for i, layer in enumerate(self.modules_LSTM):
            s, last_hidden = layer(s, hidden[i])
            final_hidden_states.append(last_hidden)
        
        # Sample
        mu = self.l2_net(s)
        sigma = torch.exp(self.l3_net(s))
        s = mu + sigma*self.N.sample(mu.shape)
        self.sampling_loss = (sigma**2 + mu**2 - torch.log(sigma)).sum()
        
        # Reshape s to (T,D)
        [N,T]=s.shape[:2]
        if VAE_ACTIVATE:
            s = s.reshape(N*T,*s.shape[2:])
        else:
            s = mu.reshape(N*T,*s.shape[2:])
        
        # Decoder output
        y = self.decoder(s)
        
        # Reshape y to (N,T,C,H,W)
        y = y.reshape(N,T,*y.shape[1:])
        
        if return_hidden:
            return y, final_hidden_states
        else:
            return y
        
    def init_states(self, N=1, value=None):
        init_hidden_states = [] # initialise hidden states
        for i in range(0,len(self.lstm_out)):
            h0 = torch.zeros((1,N,self.lstm_out[i])).to(self.device)
            c0 = torch.zeros((1,N,self.lstm_out[i])).to(self.device)
            init_hidden_states.append((h0,c0))
        return init_hidden_states