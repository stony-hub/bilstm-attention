import torch.nn
import numpy as np


class LSTM_Attention(torch.nn.Module):
    def __init__(self, voc_size, vec_dim=300, hid_dim=256, n_layers=2, drop_prob=0.5, bidirectional=True, pretrained_voc=None):
        super(LSTM_Attention, self).__init__()

        self.voc_size = voc_size
        self.vec_dim = vec_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.num_dir = 2 if bidirectional else 1
        self.n_layers = n_layers

        self.emb = torch.nn.Embedding(voc_size, vec_dim)
        self.lstm = torch.nn.LSTM(vec_dim, hid_dim, n_layers, dropout=self.drop_prob, bidirectional=bidirectional)
        self.att_linear = torch.nn.Linear(self.hid_dim * self.num_dir, self.hid_dim * self.num_dir)
        self.dis_linear_1 = torch.nn.Linear(self.hid_dim * self.num_dir, 64)
        self.dis_linear_2 = torch.nn.Linear(64, 1)
        self.softmax = torch.nn.Softmax(dim=0)
        self.sigmoid = torch.nn.Sigmoid()
        self.drp = torch.nn.Dropout(0.5)
        
        if pretrained_voc is not None:
            self.emb.from_pretrained(pretrained_voc)

    def forward(self, x, len_):
        x = self.emb(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, len_, enforce_sorted=False)
        x, (h, _) = self.lstm(x)  # [T * B * nd V] [L * nd * B * V]
        x, len_ = torch.nn.utils.rnn.pad_packed_sequence(x)  # [T * B * nd V]
        max_len, batch_size = x.shape[0], x.shape[1]
        # if self.num_dir == 2:
        #     x = x.view(max_len, batch_size, self.num_dir, self.hid_dim).transpose(-2, -1)
        #     x = x.reshape(max_len, batch_size, self.num_dir * self.hid_dim)
        h = h.view(self.n_layers, self.num_dir, batch_size, self.hid_dim)[-1,:,:,:].squeeze().transpose(0, 1).reshape(batch_size, -1)  # [B * nd V]

        att_l = self.att_linear(h)  # [B * nd V]
        att = (x * att_l).sum(dim=-1)  # [T * B]
        for i in range(batch_size):
            att[len_[i]:, i] = - np.inf
        att = self.softmax(att).unsqueeze(-1)  # [T * B * 1]

        x = (x * att).sum(dim=0)  # [B * nd V]
        x = self.dis_linear_1(x)
        x = self.drp(x)
        x = self.dis_linear_2(x)
        x = self.sigmoid(x.squeeze())  # [B]
        return x
