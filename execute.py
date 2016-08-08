#!/usr/bin/python2.7

from __future__ import print_function

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import sys
import argparse
import os
import math
import operator
import utils
import build


def generate_citation(sentence, index, search_depth=10):
    tf, df, files = index
    all_docs = set()
    for word in set(sentence):
        all_docs |= set([x for _, x in tf[word][:search_depth]])
    tfidf_sums = dict([(x, 0.) for x in all_docs])
    for word in sentence:
        idf = math.log(float(len(files)) / df[word])
        doc_tf_map = dict([(v, k) for k, v in tf[word]])
        for doc in all_docs:
            if doc in doc_tf_map:
                doc_tf = doc_tf_map[doc]
                doc_tfidf = doc_tf * idf
            else:
                doc_tfidf = 0
            tfidf_sums[doc] += doc_tfidf
    best = max(tfidf_sums.iteritems(), key=operator.itemgetter(1))
    return files[best[0]]


def generate_citations(lines, vocab, index):
    word2idx = dict([(v, k) for k, v in enumerate(vocab)])
    for line in lines[:100]:
        tokenized = list()
        capitalized = list()
        for word, cap in zip(utils.tokenize(line, periods=True), utils.tokenize(line, periods=True, capitalized=True)):
            if word == '.':
                if len(tokenized) > 10:
                    citation = generate_citation([word2idx[w] for w in tokenized if w in word2idx], index)
                    print(' '.join(capitalized) + ' (%s).' % citation)
                tokenized = list()
                capitalized = list()
            else:
                tokenized.append(word)
                capitalized.append(cap)


def main():
    parser = argparse.ArgumentParser(description='utility for building a tfidf index')
    parser.add_argument('-d', '--documents', help='path to document set')
    parser.add_argument('-v', '--vocab', help='path to vocab file (built here if missing)', default='vocab.pkl')
    parser.add_argument('-i', '--index', help='path to index file (build here if missing)', default='index.pkl')
    parser.add_argument('-f', '--file', help='the file you want to generate citations for (if none uses stdin)')

    options = parser.parse_args()
    rebuild = False # for testing

    if not os.path.exists(options.vocab) or rebuild:
        assert options.documents is not None and os.path.exists(options.documents)
        print('Could not find vocab at "%s"; building there' % options.vocab, file=sys.stderr)
        vocab = build.build_vocab(options.documents, options.vocab)
    else:
        vocab = pkl.load(open(options.vocab, 'rb'))

    if not os.path.exists(options.index) or rebuild:
        assert options.documents is not None and os.path.exists(options.documents)
        print('Could not find index at "%s"; building there' % options.index, file=sys.stderr)
        try:
            index = build.build_index(options.documents, vocab, options.index)
        except KeyError, e:
            print('Some document contained a word that was not in the vocab; regenerate the vocab to fix this.')
            raise e
    else:
        index = pkl.load(open(options.index, 'rb'))

    file = sys.stdin if options.file is None else open(options.file, 'r')
    lines = file.readlines()

    generate_citations(lines, vocab, index)


if __name__ == '__main__':
    main()
