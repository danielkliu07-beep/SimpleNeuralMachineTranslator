import math
import torch
import torch.nn as nn
from torch.nn import functional as F





#Tokenizer




class Head(nn.Module):

    #Cross attention

    def __init__(self, embed_dim, dropout):
        super().__init__()

        #query (query_dim x embed_dim)
        #key (query_dim x embed_dim)
        #value_up (query_dim x embed_dim)
        #value_down (embed_dim * query_dim)

        self.query = nn.Linear(embed_dim, embed_dim)

        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq1, seq2):
        q = self.query(seq1)

        k = self.key(seq2)
        v = self.value(seq2)

        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = F.softmax(attention, dim = -1)

        attention = self.dropout(attention)

        output = attention @ v

        return output
        


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

class FeedForward(nn.Module):

    def __init__(self, embed_dim, neuron_num, dropout):
        
        super().__init__()

        #w_up (neuron_num * embed_dim)
        #w_down (embed_dim * neuron_num)

        self.model = nn.Sequential(
            nn.Linear(embed_dim, neuron_num),
            nn.ReLU(),
            nn.Linear(neuron_num, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)


class Block(nn.Module):

    def __init__(self, embed_dim, n_head):

        super().__init__()

class Translator(nn.Module):

    def __init__(self):
        super().__init__()



#Training




