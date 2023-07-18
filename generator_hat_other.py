from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import time
import math
from torch.autograd import Variable
import numpy as np



"""
We use "NextItNet" for backbone networks.
The "ResidualBlock" class represents the network structure of each residual block.
Following the NextItNet, each residual block consists of two convolutional layers, two layer normalizations, and a ReLU activation function.
To incorporate TERACON, we introduce layer output masking for each output of the ReLU activation function.
Consequently, there are two task embeddings within each residual block, resulting in two layer output maskings for each block.

Code of NextItNet: https://github.com/syiswell/NextItNet-Pytorch
Code of CONURE: https://github.com/fajieyuan/SIGIR2021_Conure, https://github.com/yuangh-x/2022-NIPS-Tenrec
"""
class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, task_embs=[0,2]):
        super(ResidualBlock, self).__init__()
        
        """
        dilation: dilation of convolutional layer (note that the dilation of the first convolutional layer and second layer differs in accordance with NextItNet)
        kernel_size: kernel size of convolutional layer
        conv: convolutional layer
        ln: layer normalization
        ec: task embedding - The part of TERACON
        """
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        
        self.ec = nn.ModuleList([nn.Embedding(2,out_channel)])
        
        """
        The part of TERACON - Task Embedding.
        Randomly initialize the task embedding.
        lo is the lower bound of uniform distribution.
        hi is the upper bound of uniform distribution.
        """
        lo, hi = task_embs
        self.ec[0].weight.data.uniform_(lo,hi)
        self.gate = nn.Sigmoid() # Gate network for task specific masking.
        


    """
    Generate Task-Specific Masks.
    gc1: layer output masks for first layers (convolution, layer norm, activation).
    gc2: layer output masks for second layers (convolution, layer norm, activation).
    If you use more layers, you can generate for task-specific masks for each layer by adding additional task embeddings.
    """
    def mask(self, t,s): #t: task index, s: value of s
        # gc1 = self.gate(s*self.ec1(t)) #gate: sigmoid function
        # gc2 = self.gate(s*self.ec2(t))
        
        gc1 = self.gate(s*self.ec[t](torch.tensor([0]).to(t.get_device())))
        gc2 = self.gate(s*self.ec[t](torch.tensor([1]).to(t.get_device())))
        return [gc1,gc2] #Return task-specific masks.




    def forward(self, x_): 
        x = x_[0] # x: [batch_size, seq_len, embed_size].
        s = x_[1] # s: value of s.
        masks_list = x_[2] # masks_list: masks of each layers (calcuated for regularization or task similarity).
        t = x_[3] # t: index of task.
        device = x_[4] # device: gpu device.
        
        
        # Generate Task-Specific Masks. 
        masks = self.mask(torch.tensor([t]).to(device),s)
        masks_list+=masks
        gc1, gc2 = masks
        
        
        #First Layers.
        x_pad = self.conv_pad(x, self.dilation)
        out =  self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln1(out))
        
        # Apply task-specific masks (layer output masking) for first layer.
        out = out*gc1.expand_as(out)
        
        #Second Layers.
        out_pad = self.conv_pad(out, self.dilation*2)
        out = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln2(out))
        
        # Apply task-specific masks (layer output masking) for second layer.
        out = out*gc2.expand_as(out)
        
        #Residual Connection.
        out = out + x

        #Return the outputs for the next residual block
        return [out,s,masks_list, t, device]



    # Padding (code from NextItNet).
    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    
    # Truncated normal (code from NextItNet).
    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor



"""
We use "NextItNet" for backbone networks.
The "NextItNet_Decoder" class represents the network structure of backbone network (NextItNet).
To incorporate TERACON, we introduce Task-Specific Masks to backbone network.

"For the inputs of backbone network: x, s, masks_list, task_num, gpu_num, onecall."
x: input data (User behavior sequence).
s: value of s (annealing of s_max).
masks_list: a list to store the task-specific masks for each layer.
task_num: task index.
gpu_num: device (gpu) number.
onecall: if True, return the last representation of sequence.
         if False, return all representation of sequence (auto-regressive).

"For the outptus of backbone network: out, masks."
out: representation of sequence.
masks: a list of task-specific masks for each layer.

Code of NextItNet: https://github.com/syiswell/NextItNet-Pytorch
Code of CONURE: https://github.com/fajieyuan/SIGIR2021_Conure, https://github.com/yuangh-x/2022-NIPS-Tenrec
"""
class NextItNet_Decoder(nn.Module):
    def __init__(self, model_para):
        super(NextItNet_Decoder, self).__init__()
        self.model_para = model_para #Get model parameter dictionary.
        self.item_size = model_para['item_size'] # Get the total number of items.
        self.embed_size = model_para['dilated_channels'] # Get the dimension of hidden state.
        
        
        """Initialize the item embedding (Bag of item embedding)."""
        self.embeding = nn.Embedding(self.item_size, self.embed_size) 
        stdv = np.sqrt(1. / self.item_size)
        self.embeding.weight.data.uniform_(-stdv, stdv)
        
        self.task_embs = model_para['task_embs'] # Get the upper bound and lower bound of the initialization distribution for the task embedding.
        
        
        """Get hyper-parameters for convolutional layers."""
        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        
        
        """Construct networks consisting of a sequence of residual blocks."""
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation, task_embs=self.task_embs) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)


        """Construct the classication layer for task."""
        self.task_classifier = nn.ModuleList([nn.Linear(self.residual_channels, self.item_size) for n in range(1)])
        self.task_classifier[0].weight.data.normal_(0.0,0.01)
        self.task_classifier[0].bias.data.fill_(0.1)



    def forward(self, x,s,masks_list,task_num, gpu_num,onecall=False):
        """
        x: input data (User behavior sequence).
        s: value of s (annealing of s_max).
        masks_list: a list to store the task-specific masks for each layer.
        task_num: task index.
        gpu_num: device (gpu) number.
        onecall: if True, return the last representation of sequence.
                if False, return all representation of sequence.
        """
        
        
        """Get embedding of user behavior sequence from bag of item embedding"""
        inputs = self.embeding(x)
        
        
        """
        Embed the inputs to the networks.
        out_[0] = x: representation of sequence: shape [batch_size, seq_len, embed_size].
        out_[1] = s: value of s.
        out_[2] = masks_list: masks of each layers (calcuated for regularization or task similarity).
        out_[3] = t: index of task.
        out_[4] = device: gpu device.
        """
        out_ = self.residual_blocks([inputs,s,masks_list,task_num,gpu_num]) # Embed the inputs to the networks.
        dilate_outputs = out_[0] # representation of sequence.
        masks = out_[2] # A list of task-specific masks for each layer.

        if onecall: # Return the last representation of sequence.
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) 
        else: # Return all representation of sequence.
            hidden = dilate_outputs.view(-1, self.residual_channels) 
        
        out = self.task_classifier[0](hidden)


        return out, masks
