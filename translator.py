import math
import torch
import torch.nn as nn
from torch.nn import functional as F





#Tokenizer




class Head(nn.Module):

    #Cross attention

    def __init__(self, embed_dim, head_size, dropout):
        super().__init__()

        #query (query_dim x embed_dim)
        #key (query_dim x embed_dim)
        #value_up (query_dim x embed_dim)
        #value_down (embed_dim * query_dim)

        self.query = nn.Linear(embed_dim, head_size)

        self.key = nn.Linear(embed_dim, head_size)
        self.value = nn.Linear(embed_dim, head_size)

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

    def __init__(self, num_heads, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, head_size) #head_size * num_heads = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq1, seq2):
        out = torch.cat([head(seq1, seq2) for head in self.heads], dim = -1) #Finds all outputs for cross attention and turns them into a single vector
        #This vector represents all of the changes a vector embedding will take

        out = self.dropout(self.proj(out)) #Put it through a fully connected layer and apply dropout to it
        return out


class FeedForward(nn.Module):

    def __init__(self, embed_dim, dropout):
        
        super().__init__()

        #w_up (neuron_num * embed_dim)
        #w_down (embed_dim * neuron_num)
        #In this scenario, we assume neuron_num = embed_dim * 4

        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.model(x)


class Block(nn.Module):

    def __init__(self, embed_dim, n_head):

        #embed_dim = embedding dimension, n_head = number of heads
        #head_size = embedding dimensions per individual heads

        super().__init__()


        head_size = embed_dim // n_head

class Translator(nn.Module):

    def __init__(self):
        super().__init__()



#Training




