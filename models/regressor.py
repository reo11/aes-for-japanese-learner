from src.lstm import LSTM
from src.attention import Attention
import torch.nn as nn
import torch.nn.functional as F


class AttnRegressor(nn.Module):
    def __init__(self, h_dim, c_num):
        super(AttnRegressor, self).__init__()
        self.attn = Attention(h_dim)
        self.main = nn.Linear(h_dim, c_num)

    def forward(self, encoder_outputs):
        attns = self.attn(encoder_outputs)
        # (b, s, 1)
        feats = (encoder_outputs * attns).sum(dim=1)
        # (b, s, h) -> (b, h)
        return self.main(feats), attns
