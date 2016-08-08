""" Functions for building the index and vocabulary """

from __future__ import print_function

import time
import itertools
from collections import Counter
import utils
import sys
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl


def build_vocab(docs, save_as):
    start = time.time()
    vocab = set()
    for file in utils.iterate_corpus(docs):
        with open(file, 'r') as f:
            tokenized = itertools.chain.from_iterable(utils.tokenize(line) for line in f.readlines())
        vocab.update(tokenized)
    vocab = list(vocab)
    pkl.dump(vocab, open(save_as, 'wb'))
    print('Built vocabulary and saved it to "%s" in %s' % (save_as, utils.strtime(time.time() - start)), file=sys.stderr)
    return vocab


def build_index(docs, vocab, save_as):
    start = time.time()
    word2idx = dict([(v, k) for k, v in enumerate(vocab)])
    tf = dict([(i, list()) for i in xrange(len(vocab))])
    df = Counter()
    n_docs = len(list(utils.iterate_corpus(docs)))
    files = list()
    for i, file in enumerate(utils.iterate_corpus(docs)):
        print('%d/%d %s' % (i+1, n_docs, utils.strtime(time.time() - start)), file=sys.stderr, end='\r')
        files.append(file)
        with open(file, 'r') as f:
            text = f.read()
            word_counts = Counter(word2idx[w] for w in utils.tokenize(text))
            df.update(word2idx[w] for w in set(utils.tokenize(text)))
            n_words = utils.counter_sum(word_counts)
            for word, count in word_counts.items():
                tf[word].append((count / math.log(n_words), i))
    for word, docs in tf.items():
        docs.sort(key=lambda x: x[0], reverse=True)
    tfidf = tf, df, files
    pkl.dump(tfidf, open(save_as, 'wb'))
    print('Processed %d documents in %s' % (n_docs, utils.strtime(time.time() - start)), file=sys.stderr)
    return tfidf