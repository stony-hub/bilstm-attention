import os
import time
import numpy as np
import torch
from networks.lstm_emb import LSTM_Attention
from utils.data import get_IMDB_emb


class App_Emb(object):
    def __init__(self, vec_dim=300, hid_dim=100, att_dim=100, n_layers=2, drop_prob=0.5, bidirectional=True,
                 batch_size=128, data_root='data', load=True, model_root='model\\saved_emb.pth'):
        self.train_iter, self.test_iter, self.TEXT, self.LABEL = get_IMDB_emb(batch_size, data_root)
        self.model_root = model_root
        voc_size = len(self.TEXT.vocab)

        if load and os.path.exists(model_root):
            self.net = torch.load(model_root)
            print('model loaded!')
        else:
            self.net = LSTM_Attention(voc_size, vec_dim, hid_dim, att_dim, n_layers, drop_prob, bidirectional, self.TEXT.vocab.vectors).cuda()
            print('model created!')

        self.loss = torch.nn.BCELoss()
        self.opimizer = torch.optim.Adam(self.net.parameters())

    def train(self, niter=5):
        print('woohoo!')
        for epoch in range(niter):
            print('*' * 50)
            print('epoch %d / %d started' % (epoch + 1, niter))
            self.net.train()
            losses = []
            start_time = time.time()
            for i, data in enumerate(self.train_iter):
                (txt, len_), label = data.text, data.label
                self.net.zero_grad()
                out = self.net(txt.cuda(), len_)
                loss = self.loss(out, (label - 1).float().cuda())
                loss.backward()
                self.opimizer.step()

                losses.append(loss.item())

            used_time = time.time() - start_time

            print('epoch %d / %d finished, time: %.2f, loss %.5f' % (epoch + 1, niter, used_time, np.array(losses, dtype=np.float).mean()))

            torch.save(self.net, self.model_root)

            self.test()

    def test(self):
        self.net.eval()
        total, correct = 0, 0
        pp, pn, np, nn = 0, 0, 0, 0
        for data in self.test_iter:
            (txt, len_), label = data.text, data.label
            out = self.net(txt.cuda(), len_).cpu()
            crr = 0
            for i, x in enumerate(out):
                t2, t1 = x > 0.5, label[i] == 2
                if t1 == t2: crr += 1
                if t1 and t2: pp += 1
                if t1 and not t2: pn += 1
                if not t1 and t2: np += 1
                if not t1 and not t2: nn += 1
            correct += crr
            total += len(label)
        print(correct, total)
        print(pp, pn, np, nn)
        print('evaluated, acc=%.3f' % (correct / total * 100))

    def predict(self, str_):
        torch.cuda.empty_cache()
        self.net.eval()
        input_ = torch.zeros(len(str_), 1, dtype=torch.long)
        for i, x in enumerate(str_):
            input_[i, 0] = self.TEXT.vocab.stoi[x]
        len_ = torch.tensor([len(str_)], dtype=torch.long)
        output = self.net(input_.cuda(), len_)
        return output.item()
