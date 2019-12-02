import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# attention layer code inspired from:
# https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
# https://qiita.com/itok_msi/items/ad95425b6773985ef959


class Attention(nn.Module):
    def __init__(self, h_dim):
        super(Attention, self).__init__()
        self.h_dim = h_dim
        self.fc1 = nn.Linear(h_dim, 24)
        self.fc2 = nn.Linear(24, 1)

    def forward(self, inputs):
        b_size = inputs.size(0)
        x = inputs.contiguous().view(-1, self.h_dim)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.fc2(x))
        out = F.softmax(x.contiguous().view(b_size, -1), dim=1).unsqueeze(2)
        return out
