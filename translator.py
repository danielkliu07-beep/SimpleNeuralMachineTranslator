import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from torchtext.datasets import Multi30k



#Tokenizer


#Positional Embedding

class PositionEncoding(nn.Module):

    def __init__(self, d_model = 2, max_len = 6): #d_model - dimension of the model (number of word embedding values per token), max_len - max number of tokens our Transformer can process (input and output combined)

        super().__init__()

        pe = torch.zeros(max_len, d_model) #6 rows by 2 columns matrix of zeros

        position = torch.arange(start = 0, end = max_len, step = 1).float().unsqueeze(1) #creates a tensor of [0, 1, 2, 3, 4, 5], .unsqueeze(1) turns the tensor into a column matrix
        embedding_index = torch.arange(start = 0, end = d_model, step = 2).float() #tensor of [0, 2], same as 2i

        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model) #tensor of all 10000^(2i/d_model))

        pe[:, 0::2] = torch.sin(position * div_term) #Applies sin to even columns - 'start:stop:step -> 0::2'
        pe[:, 1::2] = torch.cos(position * div_term) #Applies sin to odd columns

        self.register_buffer('pe', pe) #Ensures pe gets moved to a GPU if we use one 
    
    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :] #Adds positional encoding values to the word embedding values



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
    
    def forward(self, seq1, seq2, mask = None):
        q = self.query(seq1)

        k = self.key(seq2)
        v = self.value(seq2)

        attention = 0

        if mask is None:
            attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attention = F.softmax(attention, dim = -1)

            attention = self.dropout(attention)
        else:
            attention = (q @ k.transpose(-2, -1)) * (k.shape[-1]**-0.5)
            attention = attention.masked_fill(mask = mask, value = float('-inf'))

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


class EncoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        #Encoder block goes through multiheaded attention, layer norm, and feed forward

        head_size = embed_dim // num_heads
        self.cross_attention = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        return x

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        #Decoder block goes through masked multi-headed attention, multi-headed attention, layer norm, and feed forward




class Translator(nn.Module):

    def __init__(self):
        super().__init__()

        #Take in input
        #Position encode input
        #Put the input through multi-headed attnetion
        #Feed forward input





#Training




