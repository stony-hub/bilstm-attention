import re
import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(x: list) -> list:
    review = ' '.join(x)
    review = REPLACE_WITH_SPACE.sub(' ', review).lower()
    tokens = review.split()
    tokens = list(map(lambda a: lemmatizer.lemmatize(a, 'v'), tokens))
    tokens = list(filter(lambda a: a not in stop_words, tokens))
    return tokens


def get_IMDB(batch_size=128, root='data'):
    print('start getting data!')
    print('step 1/5 set up fields')
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=False)
    LABEL = data.Field(sequential=False)

    print('step 2/5 get data sets')
    train, test = datasets.IMDB.splits(TEXT, LABEL, root=root)

    print('step 3/5 preprocess data')
    for x in train.examples:
        x.text = preprocess(x.text)
    for x in test.examples:
        x.text = preprocess(x.text)

    print('step 4/5 build the vocabulary')
    TEXT.build_vocab(train)
    LABEL.build_vocab(train)

    print('step 5/5 make iterators')
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size)
    print('finish getting data!')
    return train_iter, test_iter, TEXT, LABEL


def get_IMDB_emb(batch_size=128, root='data'):
    print('start getting data!')
    print('step 1/5 set up fields')
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=False)
    LABEL = data.Field(sequential=False)

    print('step 2/5 get data sets')
    train, test = datasets.IMDB.splits(TEXT, LABEL, root=root)

    print('step 3/5 preprocess data')
    for x in train.examples:
        x.text = preprocess(x.text)
    for x in test.examples:
        x.text = preprocess(x.text)

    print('step 4/5 build the vocabulary')
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))
    LABEL.build_vocab(train)

    print('step 5/5 make iterators')
    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size)
    print('finish getting data!')
    return train_iter, test_iter, TEXT, LABEL
