""" Miscellaneous utility functions """

import os
import re
from collections import Counter


def iterate_corpus(docs):
    for subdir, dirs, files in os.walk(docs):
        for file in files:
            yield os.path.join(subdir, file)


def tokenize(line, periods=False, capitalized=False):
    return [w if capitalized else w.lower() for w in re.findall('[a-zA-Z0-9]+' + ('|\.' if periods else ''), line)]


def strtime(t):
    m, s = divmod(t, 60)
    if m > 0:
        h, m = divmod(m, 60)
        if h > 0:
            return '%dm %dm %ds' % (h, m, s)
        else:
            return '%dm %ds' % (m, s)
    else:
        return '%ds' % s


def inv_dict(d):
    assert isinstance(d, dict)
    return dict([(v, k) for k, v in d.items()])


def counter_sum(c):
    assert isinstance(c, Counter)
    return sum([x for _, x in c.items()])
