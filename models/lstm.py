import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import math
import numpy as np
from src.attention import Attention
from torch.autograd import Variable

torch.manual_seed(1)
# device = torch.device('cpu')
# https://www.kaggle.com/dannykliu/lstm-with-attention-clr-in-pytorch


class LSTM(nn.Module):
    def __init__(self, hidden_dim, embedding_matrix, device="cuda", batch_first=True):
        super(LSTM, self).__init__()
        self.device = device
        self.hid_dim = hidden_dim
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.word_embeddings.load_state_dict(
            {'weight': torch.tensor(embedding_matrix).to(self.device)})
        self.lstm = nn.LSTM(embedding_dim, self.hid_dim,
                            batch_first=batch_first,
                            dropout=0.5)

    def init_hidden(self, b_size):
        h0 = Variable(torch.zeros(1, b_size, self.hid_dim)).to(self.device)
        c0 = Variable(torch.zeros(1, b_size, self.hid_dim)).to(self.device)
        return (h0, c0)

    def forward(self, sentence, lengths=None):
        self.hidden = self.init_hidden(sentence.size(0))
        emb = self.word_embeddings(sentence.long())
        out, hidden = self.lstm(emb, self.hidden)
        return out
