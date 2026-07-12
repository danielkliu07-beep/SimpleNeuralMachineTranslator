import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from torchtext.datasets import Multi30k



#Tokenizer


#Positional Embedding

class PositionEncoding(nn.Module):

    def __init__(self, d_model, max_len): 

        super().__init__()

        pe = torch.zeros(max_len, d_model).unsqueeze(0)

        position = torch.arange(start = 0, end = max_len, step = 1).float().unsqueeze(1) 
        embedding_index = torch.arange(start = 0, end = d_model, step = 2).float()

        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        self.register_buffer('pe', pe) 

    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :] 



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

    def __init__(self, num_heads, head_size, embed_dim, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_dim, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embed_dim) #head_size * num_heads = embed_dim
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq1, seq2, mask = None):
        out = torch.cat([head(seq1, seq2, mask) for head in self.heads], dim = -1) #Finds all outputs for cross attention and turns them into a single vector
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
        self.sa = MultiHeadAttention(num_heads, head_size, embed_dim, dropout)
        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, padding_mask = None):

        #input -> Normalize input -> put it through sa -> add this to itself
        #input -> Normalize input -> put it through ffwd -> add it to itself

        x = x + self.sa(self.ln1(x), self.ln1(x), mask = padding_mask)
        x = x + self.ffwd(self.ln2(x))

        return x

class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        #Decoder block goes through masked multi-headed attention, multi-headed attention, layer norm, and feed forward

        head_size = embed_dim // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, embed_dim, dropout)
        self.ca = MultiHeadAttention(num_heads, head_size, embed_dim, dropout)

        self.ffwd = FeedForward(embed_dim, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim) #layer norm 2 
        self.lnenc2 = nn.LayerNorm(embed_dim) #layer norm 2 for encoder outputs
        self.ln3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, enc_kv, causal_mask = None, padding_mask = None): #(x, encoder_kv) -> x is input, encoder_kv -> outputs of encoder

        #causal mask - hides future info, prevents it from looking ahead
        #padding mask - hides blank spaces for cross-attention since they're useless, completely optional

        x = x + self.sa(self.ln1(x), self.ln1(x), mask = causal_mask)

        x = x + self.ca(self.ln2(x), self.lnenc2(enc_kv), mask = padding_mask)

        x = x + self.ffwd(self.ln3(x))

        return x




class Translator(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_heads, num_layers, dropout, max_len = 5000):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoding = PositionEncoding(embed_dim, max_len)

        #Stacking multiple blocks (layers) using nn.ModuleList
        self.encoder_blocks = nn.ModuleList([EncoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)])

        self.fc_layer = nn.Linear(in_features = embed_dim, out_features = tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask = None, tgt_mask = None):

        #Encoder Block

        src_embedding = self.src_embedding(src)
        src_positional_encoding = src_embedding + self.pos_encoding(src_embedding)

        encoder_outputs = torch.cat([EncoderBlock(src_positional_encoding, src_mask) for EncoderBlock in self.encoder_blocks])

        #Decoder Block

        tgt_embedding = self.tgt_embedding(tgt)
        tgt_positional_encoding = tgt_embedding + self.pos_encoding(tgt_embedding)

        decoder_outputs = torch.cat([DecoderBlock(tgt_positional_encoding, encoder_outputs, src_mask, tgt_mask) for DecoderBlock in self.decoder_blocks])

        #Final output

        output = self.fc_layer(decoder_outputs)

        output = F.softmax(output, dim = -1)

        return output



#Training




