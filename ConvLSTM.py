"""Code obtained by Stefano Pini and Andrea Palazzi"""

import torch.nn as nn
import torch
from constants import *


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (h,c)


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is None:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        else:
            # Prepare data by detaching it
            #self._detach_hidden(hidden_state)
            pass

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
    
    def _detach_hidden(self, hidden_state):
        for (h,c) in hidden_state:
            h.detach_()
            c.detach_()
            

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
class ModelCLSTM(nn.Module):
    def __init__(self, kernel_size, cnn_dim, clstm_hidden_dim):
        super().__init__()
        
        self.padding = kernel_size // 2, kernel_size // 2
        
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding='same'),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=1,stride=1),
            nn.ReLU()
            )
        
        self.clstm_layer = ConvLSTM(input_dim=256,
                 hidden_dim=[256,512,512,256],
                 kernel_size=[(3,3),(3,3),(3,3),(1,1)],
                 num_layers=4,
                 batch_first=True,
                 bias=True,
                 return_all_layers=True)
            
        self.tcnn_layer = nn.Sequential(
            nn.ConvTranspose2d(256,64,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,kernel_size=1)
        )
        
    def forward(self, x, hidden= None, return_hidden = False):
        # Define x with shape (T,C,H,W)
        x = x.squeeze(0)
        s1 = self.cnn_layer(x) # s - static output
        
        # Reshape s to (1,T,C,H,W)
        s1 = s1.unsqueeze(0)
        
        # ConvLSTM output
        s2, hidden = self.clstm_layer(s1, hidden)
        s2 = s2[-1]
        s2 = s2.squeeze(0)
        
        # CNN output
        y = self.tcnn_layer(s2)
        y = y.unsqueeze(0)
        
        if return_hidden:
            return y, hidden
        else:
            return y
    
    def init_states(self, value=None):
        states = self.clstm_layer._init_hidden(batch_size=1, image_size=(4, 4))
        return states
    
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
        
        # self.cnn_3D = nn.Sequential(
        #     nn.Conv3d(16,1,kernel_size=3,padding=1),
        #     nn.Sigmoid()
        # )
        
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
        
        # Pass through Conv3d for smoothing
        #y = torch.permute(y, [0,2,1,3,4])
        #y = self.cnn_3D(y)
        #y = torch.permute(y, [0,2,1,3,4])
        
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