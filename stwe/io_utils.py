import numpy as np
import time
from datetime import datetime
import operator
from gensim.models import Word2Vec
from .utils import get_logger

log = get_logger("io_utils")


class DocumentTimePair(object):
    def __init__(self, path, time_transform=0, start_time=0, s_prob=1):
        self.path = path
        self.preproc = Preprocessing()
        self.time_transform = time_transform
        self.start_time = start_time
        self.s_prob = s_prob

    def __iter__(self):
        for line in self.path:
            if self.s_prob == 1 or self.s_prob > np.random.rand():
                parts = line.split("\t")
                if self.time_transform != 0 and self.start_time != 0:
                    parts[0] = (float(parts[0]) - self.start_time) / self.time_transform
                try:
                    yield ( float(parts[0]), self.preproc.pp_doc(parts[1]) )
                except:
                    continue


def read_clst_file(in_file):
    clst = []
    with open(in_file) as data:
        for line in data:
            clst.append(int(line))
    return clst



def sort_hashtags(in_file, out_file):
    dict = {}
    with open(in_file) as data:
        for line in data:
            for tag in line.split():
                if tag not in dict:
                    dict[tag] = 1
                else:
                    dict[tag] += 1

    dict_sorted = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    out = open(out_file, 'w')
    for one in dict_sorted:
        out.write("{}\t{}\n".format(one[0], one[1]))

    out.close()

def read_cent_repr(in_file):
    """Returns a dictionary of word-vector pairs.
    This pairs are used as central representations in the model. 
    
    'in_file' - file containing word-vector pairs.
    """
    reprs = {}
    w2v = Word2Vec.load(in_file)
    for k in w2v.vocab:
         reprs[k] = w2v.syn0[w2v.vocab[k].index]

    """
    reprs = {}
    with open(in_file) as data:
        for line in data:
            name_repr = line.split("\t")
            parts = name_repr[1].strip().split()
            vec = array([float(x) for x in parts])
            reprs[name_repr[0].strip(" \t")] = vec

    return reprs
    """
    return reprs
