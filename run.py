import os
import re
import numpy as np
from approaches.lstm_emb import App_Emb
from utils.data import preprocess


def to_predict(app):
    while True:
        print('输入一个句子')
        l = input()
        l = preprocess(' '.join(l))
        if l != []:
            print(app.predict(l))
        else:
            print('oh...')


def do_test(app, path, res):
    score = []
    lst = list(os.walk(path))[0][2]
    for i, p in enumerate(lst):
        sc = []
        with open(path + p, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            num = len(lines)
            for l in lines:
                l = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', l)
                l = preprocess(' '.join(l))
                if l != []:
                    sc.append(app.predict(l))
            if len(sc) != 0:
                num = len(sc)
                sc = np.array(sc)
                score.append([p, num, sc.mean(), sc.std()])
        print('%s / %s finished' % (i+1, len(lst)))
    score.sort(key=lambda x: x[2], reverse=True)
    with open(res, 'w') as f:
        for p, n, m, s in score:
            f.write('%s : %d records, score : %.3f , std : %.3f' % (p, n, m, s) + '\n')

def do_rec(app, path):
    res_path = os.path.join(path, 'res')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    score = []
    lst = list(os.walk(path))[0][2]
    for i, p in enumerate(lst):
        sc = []
        with open(path + p, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            num = len(lines)
            for l in lines:
                l = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', l)
                l = preprocess(' '.join(l))
                if l != []:
                    sc.append(app.predict(l))
        with open(os.path.join(res_path, p), 'w', encoding='UTF-8') as f:
            for x in sc:
                f.write(str(x) + '\n')
        print('%s / %s finished' % (i+1, len(lst)))
    score.sort(key=lambda x: x[2], reverse=True)

def do_test_rate(app, path, res, thre1=0.3, thre2=0.7):
    score = []
    lst = list(os.walk(path))[0][2]
    for i, p in enumerate(lst):
        good, bad = 0, 0
        with open(path + p, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for l in lines:
                l = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', l)
                l = preprocess(' '.join(l))
                if l != []:
                    sc = app.predict(l)
                    if sc >= thre2: good += 1
                    elif sc <= thre1: bad += 1
            if good + bad != 0:
                score.append([p, good / (good + bad), good, bad])
        print('%s / %s finished' % (i+1, len(lst)))
    score.sort(key=lambda x: x[1], reverse=True)
    with open(res, 'w') as f:
        for p, s, n1, n2 in score:
            f.write('%s rate: %.3f, good: %d, bad: %d' % (p, s, n1, n2) + '\n')


def main():
    app = App_Emb(vec_dim=100, hid_dim=256, att_dim=128, batch_size=64, load=True)
    # app.test()
    # app.train(niter=0)
    # do_rec(app, 'data\\new\\')
    to_predict(app)


if __name__ == '__main__':
    main()
