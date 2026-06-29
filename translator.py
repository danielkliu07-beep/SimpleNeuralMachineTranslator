import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import TensorDataset, DataLoader

#English -> Chinese


class Attention(nn.Module):

    def __init__(self, d_model = 2):

        super().__init__()

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False) 
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False) 
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False) 

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q) 
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1 = self.col_dim))
        scaled_sims = sims / (k.size(self.col_dim) ** 0.5)

        if mask is not None: 
            scaled_sims = scaled_sims.masked_fill(mask = mask, value = -1e9) 

        attention_percents = F.softmax(scaled_sims, dim = self.col_dim) 

        attention_scores = torch.matmul(attention_percents, v) 

        return attention_scores

class EncoderRnn(nn.Module):

    def __init__(self, num_tokens, d_model, max_len):

        super().__init__()

        self.we = nn.Embedding(num_embeddings = num_tokens, embedding_dim = d_model)

        self.bi_lstm = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1, bidirectional=True)
        
        self.uni_lstm1 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm2 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm3 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm4 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm5 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm6 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.uni_lstm7 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)

    def forward(self, token_id):

        word_embedding1 = self.we(token_id)
        bi_lstm_out = self.bi_lstm(word_embedding1)

        uni_lstm_out1 = self.uni_lstm1(bi_lstm_out)
    
        uni_lstm_out2 = self.uni_lstm2(uni_lstm_out1)
        residual_connection_value1 = uni_lstm_out2 + uni_lstm_out1

        uni_lstm_out3 = self.uni_lstm3(residual_connection_value1)
        residual_connection_value2 = residual_connection_value1 + uni_lstm_out3

        uni_lstm_out4 = self.uni_lstm4(residual_connection_value2)
        residual_connection_value3 = residual_connection_value2 + uni_lstm_out4

        uni_lstm_out5 = self.uni_lstm5(residual_connection_value3)
        residual_connection_value4 = residual_connection_value3 + uni_lstm_out5
        

        uni_lstm_out6 = self.uni_lstm6(residual_connection_value4)
        residual_connection_value5 = residual_connection_value4 + uni_lstm_out6
        

        uni_lstm_out7, temp = self.uni_lstm7(residual_connection_value5)
        residual_connection_value6 = residual_connection_value5 + uni_lstm_out7

        return residual_connection_value6
        




class DecoderRnn(nn.Module):

    def __init__(self, num_tokens, d_model, max_len):

        super().__init__()

        self.lstm1 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm2 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm3 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm4 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm5 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm6 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm7 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
        self.lstm8 = nn.LSTM(input_size = 1, hidden_size = 1, num_layers = 1)
