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

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, task_embs=[0,2],num_task=1):
        super(ResidualBlock, self).__init__()
        
        """
        dilation: dilation of convolutional layer (note that the dilation of the first convolutional layer and second layer differs in accordance with NextItNet)
        kernel_size: kernel size of convolutional layer
        num_task: Total number of tasks
        conv: convolutional layer
        ln: layer normalization
        ec: task embedding - The part of TERACON
        """
        self.dilation = dilation
        self.kernel_size = kernel_size
        
        self.num_task = num_task
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        """
        The part of TERACON - Task Embedding.
        Randomly initialize the task embedding.
        lo is the lower bound of uniform distribution.
        hi is the upper bound of uniform distribution.
        Generate "n" task embeddings (where "n" is the total number of tasks).
        """ 
        lo, hi = task_embs
        self.ec = nn.ModuleList([nn.Embedding(2,out_channel) for n in range(num_task)])
        for n in range(num_task):
            self.ec[n].weight.data.uniform_(lo,hi)
        self.gate = nn.Sigmoid() # Gate network for task specific masking.
        self.tanh = nn.Tanh() # Gate network for relation-aware
        
        """
        The part of TERACON - Task Relation Aware.
        As we apply task masks to two layers within each residual block, two MLPs are generated for each residual block.
        If there are additional layers, you can add more MLP layers accordingly.
        """
        self.mlp1 = nn.ModuleList(nn.Linear(2*num_task-1,1) for i in range(num_task))
        self.mlp2 = nn.ModuleList(nn.Linear(2*num_task-1,1) for i in range(num_task))
        
        for i in range(num_task): # Initialize the task relation aware mlp
            self.mlp1[i].weight.data.normal_(0.0,0.01)
            self.mlp1[i].bias.data.fill_(0.1)
            self.mlp2[i].weight.data.normal_(0.0,0.01)
            self.mlp2[i].bias.data.fill_(0.1)

    def mask(self, t,s):
        gc1 = self.gate(s*self.ec[t](torch.tensor([0]).to(t.get_device())))
        gc2 = self.gate(s*self.ec[t](torch.tensor([1]).to(t.get_device())))
        return [gc1,gc2]
    
    
    
    """
    Generate Relation-Aware Task-Specific Masks.
    gc1: layer output masks for first layers (convolution, layer norm, activation).
    gc2: layer output masks for second layers (convolution, layer norm, activation).
    If you use more layers, you can generate for task-specific masks for each layer by adding additional task embeddings.
    
    We concatenate current task and other tasks (positive task embedding and negative task embedding) and apply mlp for relation aware.
    Refer Equations 12 and 13 in the paper.
    """
    def attn_fmask(self,t,s):
        new_task_emb_1 = self.ec[t](torch.tensor([0]).to(t.get_device())) # Extract current task embdding.

        for i in range(0,self.num_task): # Concatenate current task embedding and other task embeddings (positive and negative).
            if not i ==t:
                new_task_emb_1 = torch.cat((new_task_emb_1,self.ec[i](torch.tensor([0]).to(t.get_device()))))
                new_task_emb_1 = torch.cat((new_task_emb_1,-1*self.ec[i](torch.tensor([0]).to(t.get_device()))))

        new_task_emb_2 = self.ec[t](torch.tensor([1]).to(t.get_device())) # Extract current task embdding.
        for i in range(0,self.num_task): # Concatenate current task embedding and other task embeddings (positive and negative).
            if not i ==t:
                new_task_emb_2 = torch.cat((new_task_emb_2,self.ec[i](torch.tensor([1]).to(t.get_device()))))
                new_task_emb_2 = torch.cat((new_task_emb_2,-1*self.ec[i](torch.tensor([1]).to(t.get_device()))))

        # Relation aware using mlp. Refer Equation 12 in the paper.
        attn_output_1 = self.mlp1[t](self.tanh(s*new_task_emb_1).transpose(1,0)).transpose(1,0)
        attn_output_2 = self.mlp2[t](self.tanh(s*new_task_emb_2).transpose(1,0)).transpose(1,0)
        gc1 = self.gate(s*attn_output_1)
        gc2 = self.gate(s*attn_output_2)
        return [gc1,gc2]

    def forward(self, x_): 
        x = x_[0] # x: [batch_size, seq_len, embed_size].
        s = x_[1] # s: value of s.
        masks_list = x_[2] # masks_list: masks of each layers (calcuated for regularization or task similarity).
        t_ = x_[3] # t: index of task.
        device = x_[4] # device: gpu device.
        backward = x_[5] # backward: The indicator for whether the task was in the past is used only when learning task 2 from task 1.
        new_task = x_[6] # new_task: Index of the newly emerged task.
        
        
        t = torch.tensor([t_]).to(device)


        # Generate Task Masks. 
        if t_ ==new_task: # If it is newly emerged task, conduct relation aware with other tasks.
            masks = self.attn_fmask(t,s)
        else:
            """
            When learning from task 1 to task 2, if the backward flag is set to False, it indicates that the pseudo label generation is for task 1.
            After training task 2, we can obtain relation-aware task masks for task 1, and we no longer use the self.mask() function.
            Consequently, after task 2, we can generate pseudo labels for each task using the relation-aware task masks without relying on the self.mask() function.
            """
            if (not new_task ==1) or backward:
                masks = self.attn_fmask(t,s) # Relation aware task masks
            else:
                """
                To make pseudo labels (knowlege retention) of task 1 soley (when learn task 1 to task 2), use self.mask function to make mask of task1.
                After training task 2, we can obtain relation-aware task masks for task 1, and we no longer use the self.mask() function.
                """
                masks = self.mask(t,s) 
        masks_list+=masks
        gc1, gc2 = masks
        
        
        #First Layers.
        x_pad = self.conv_pad(x, self.dilation)
        out =  self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln1(out))
        
        # Apply relation-aware task-specific masks (layer output masking) for first layer.
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
        return [out,s,masks_list, t_, device,backward,new_task]



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
backward: The indicator for whether the task was in the past is used only when learning task 2 from task 1.
new_task: Index of the newly emerged task.

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
        stdv = np.sqrt(1. / self.item_size) # Because we load the item embedding, we don't need to initialize the item embedding
        self.embeding.weight.data.uniform_(-stdv, stdv)


        self.task_embs = model_para['task_embs'] # Get the upper bound and lower bound of the initialization distribution for the task embedding.
        self.num_task = model_para['num_task'] # Get the total number of tasks.
        self.target_size = model_para['target_item_size'] # Get the list of the total number of past tasks' target


        """Get hyper-parameters for convolutional layers."""
        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        
        
        """Construct networks consisting of a sequence of residual blocks."""
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation, task_embs=self.task_embs,num_task=self.num_task) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)
        
        """Construct the classication layer for each tasks (current task and past tasks)."""
        self.task_classifier = nn.ModuleList([nn.Linear(self.residual_channels, self.target_size[n]) for n in range(self.num_task)])
        for i in range(self.num_task): # Initialize the classifier
            self.task_classifier[i].weight.data.normal_(0.0,0.01)
            self.task_classifier[i].bias.data.fill_(0.1)
        
        
    def forward(self, x,s,masks_list,task_num, gpu_num,onecall=False,backward=False,new_task=2): # inputs: [batch_size, seq_len]
        """
        x: input data (User behavior sequence).
        s: value of s (annealing of s_max).
        masks_list: a list to store the task-specific masks for each layer.
        task_num: task index.
        gpu_num: device (gpu) number.
        onecall: if True, return the last representation of sequence.
                if False, return all representation of sequence (auto-regressive).
        backward: The indicator for whether the task was in the past is used only when learning task 2 from task 1.
        new_task: Index of the newly emerged task.
        """
        

        """Get embedding of user behavior sequence from bag of item embedding"""
        inputs = self.embeding(x) # [batch_size, seq_len, embed_size]
        
        
        """
        Embed the inputs to the networks.
        out_[0] = x: representation of sequence: shape [batch_size, seq_len, embed_size].
        out_[1] = s: value of s.
        out_[2] = masks_list: masks of each layers (calcuated for regularization or task similarity).
        out_[3] = t: index of task.
        out_[4] = device: gpu device.
        out_[5] = backward: The indicator for whether the task was in the past is used only when learning task 2 from task 1.
        out_[6]  = new_task: Index of the newly emerged task.
        """
        out_ = self.residual_blocks([inputs,s,masks_list,task_num,gpu_num, backward,new_task]) # Embed the inputs to the networks.
        dilate_outputs = out_[0] # representation of sequence.
        masks_ = out_[2] # A list of task-specific masks for each layer.

        if onecall: # Return the last representation of sequence.
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) # [batch_size, embed_size]
        else: # Return all representation of sequence.
            hidden = dilate_outputs.view(-1, self.residual_channels) # [batch_size*seq_len, embed_size] 

        out = self.task_classifier[task_num](hidden)

        return out, masks_
