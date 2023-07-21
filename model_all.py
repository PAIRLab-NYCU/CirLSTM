__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np

from CausalLSTM import CausalLSTMCell
from GradientHighwayUnit import GHU

from utils.pixelShuffle_torch import pixel_shuffle

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        
        num_hidden = [int(x) for x in args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        print("> LSTM layers: {}".format(num_layers))
        print("> LSTM each layer's hidden number: {}".format(num_hidden))
        self.num_layers = num_layers
        
        self.seq_length = args.seq_length
        self.frame_channel = args.patch_size * args.patch_size * args.channel
        self.origin_frame_channel = args.patch_size * args.patch_size * 1
        self.patch_size = args.patch_size
        self.num_hidden = num_hidden
        self.lost_interval = args.lost_interval
        
        lstm_fw = []
        lstm_bw = []
        h = args.h // args.patch_size
        w = args.w // args.patch_size
        # initialize lstm architecture
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            lstm_fw.append(
                CausalLSTMCell(in_channel, num_hidden[i], h, w, args.filter_size, args.stride)
            )
            lstm_bw.append(
                CausalLSTMCell(in_channel, num_hidden[i], h, w, args.filter_size, args.stride)
            )
        self.lstm_fw = nn.ModuleList(lstm_fw)
        self.lstm_bw = nn.ModuleList(lstm_bw)
        
        # Initialize GHU unit
        self.ghu = GHU(in_channel, num_hidden[0], h, w, args.filter_size, args.stride)
        
        hidden_concat_conv = []
        mem_concat_conv = []
        for l in range(num_layers):
            hidden_concat_conv.append(
                nn.Conv2d(num_hidden[l]*2, num_hidden[0], kernel_size=1, stride=1, padding=0)
            )
            mem_concat_conv.append(
                nn.Conv2d(num_hidden[l]*2, num_hidden[0], kernel_size=1, stride=1, padding=0)
            )
        self.hidden_concat_conv = nn.ModuleList(hidden_concat_conv)
        self.mem_concat_conv = nn.ModuleList(mem_concat_conv)
            
        # initialize generate convolution
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.origin_frame_channel, kernel_size=1, stride=1, padding=0)
            
    def zero(self, batch, height, width):
        return torch.zeros([batch, self.num_hidden[0], height, width]).type(torch.cuda.FloatTensor)
        
        
    def forward(self, fw_seq, bw_seq, origin_seq):
        # Initialize parameter
        batch = fw_seq.shape[0]
        height = fw_seq.shape[3]
        width = fw_seq.shape[4]
        
        # Initialize LSTM hidden state and cell state
        hidden_fw = [self.zero(batch, height, width) for _ in range(self.num_layers)]
        hidden_bw = [self.zero(batch, height, width) for _ in range(self.num_layers)]
        cell_fw = [self.zero(batch, height, width) for _ in range(self.num_layers)]
        cell_bw = [self.zero(batch, height, width) for _ in range(self.num_layers)]
        
        tm_hidden_fw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_hidden_bw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_mem_fw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_mem_bw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        
        gen_images = []
        
        # Start LSTM pass
        for l in range(self.num_layers):
            for t in range(self.seq_length):
                if l == 0:  # if first layer
                    inputs_fw = fw_seq[:, t]
                    inputs_bw = bw_seq[:, t]
                    memory_fw = self.zero(batch, height, width)
                    memory_bw = self.zero(batch, height, width)
                else:
                    inputs_fw = hiddenConcated_lst[t]
                    inputs_bw = hiddenConcated_lst[self.seq_length-t-1]
                    memory_fw = memConcated_lst[t]
                    memory_bw = memConcated_lst[self.seq_length-t-1]
                    
                # forward
                hidden_fw[l], cell_fw[l], memory_fw = self.lstm_fw[l](inputs_fw, hidden_fw[l], cell_fw[l], memory_fw)
                # backward
                hidden_bw[l], cell_bw[l], memory_bw = self.lstm_bw[l](inputs_bw, hidden_bw[l], cell_bw[l], memory_bw)

                # GHU, only first layer need to use GHU
                if l == 0:
                    hidden_fw[l] = self.ghu(hidden_fw[l], self.zero(batch, height, width))
                    hidden_bw[l] = self.ghu(hidden_bw[l], self.zero(batch, height, width))

                # save forward and backward output
                # forward
                tm_hidden_fw[l][t] = hidden_fw[l]
                tm_mem_fw[l][t] = memory_fw
                # backward
                tm_hidden_bw[l][t] = hidden_bw[l]
                tm_mem_bw[l][t] = memory_bw

            hiddenConcated_lst = [None for i in range(self.seq_length)]
            memConcated_lst = [None for i in range(self.seq_length)]
            for t in range(self.seq_length):
                # Concatenate
                hiddenConcat = torch.cat([tm_hidden_fw[l][t], tm_hidden_bw[l][self.seq_length-1-t]], axis=1)
                memConcat = torch.cat([tm_mem_fw[l][t], tm_mem_bw[l][self.seq_length-1-t]], axis=1)

                # Convolve back to origin channel
                hiddenConcated_lst[t] = self.hidden_concat_conv[l](hiddenConcat)
                memConcated_lst[t] = self.mem_concat_conv[l](memConcat)
                
        x_gen = [None for i in range(self.seq_length)]
        # Generate complete output
        for t in range(self.seq_length):
            if t % self.lost_interval == 0:
                gen = origin_seq[:, t]
                x_gen[t] = pixel_shuffle(gen, self.patch_size)
            else:
                gen = self.conv_last(hiddenConcated_lst[t])
                x_gen[t] = pixel_shuffle(gen, self.patch_size)
                
        pred_frames = torch.stack(x_gen, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        
        return pred_frames