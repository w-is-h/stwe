from io_utils import DocumentTimePair
import numpy as np
from itertools import permutations

def get_distr():
    ttp = DocumentTimePair("/local/kraljevic/twiter/processed/1M_train.dat")

    dict = {}
    start = None
    day = 0

    tags_f = open("/local/kraljevic/twiter/tags/1M_train.dat")

    for tweet in ttp:
        if start is None:
            #Get time for start
            start = int(tweet[0])
        #3600 * 24 for days
        if int((int(tweet[0]) - start) / (3600 * 24)) > day:
            day += 1

        tags = tags_f.readline().strip()
        
        for tag in tags.split(" "):
            if tag != "__NONE__":
                if tag in dict:
                    dict[tag][day] += 1
                else:
                    dict[tag] = np.zeros(220)
                    dict[tag][day] += 1

    return dict


def get_tags_my():
    dict = get_distr()
    tags = []

    for key in dict.keys():
        if np.sum(dict[key]) < 100:
            continue

        max_one = max(dict[key])
        cnt = 0
        for one in dict[key]:
            if one != 0:
                if max_one / one < 10:
                    cnt += 1

        if cnt < 8:
            tags.append(key)

    return (tags, dict)


def get_tags_gini():
    dict = get_distr()
    tags = []
    for key in dict.keys():
        if np.sum(dict[key]) < 50:
            continue
        s = dict[key]
        k = 0
        for i, j in list(permutations(s,2)):
            k += abs(i-j)
        MD = k/float(len(s)**2)
        G = MD / float(np.mean(s))
        G = G/2
        if G > 0.85:
            tags.append(key)

    return (tags, dict)

