from nltk.corpus import stopwords
import codecs
import numpy as np
import time
from datetime import datetime
import re
import operator
from gensim.models import Word2Vec

class word2vecIterator(object):
    def __init__(self, path):
        self.path = path
        self.preproc = Preprocessing()

    def __iter__(self):
        for line in open(self.path):
            parts = line.split("\t")
            yield [x.strip() for x in parts[1].split(" ")]

class DocumentTimePair(object):
    def __init__(self, path, time_transform=0, start_time=0, s_prob=1):
        self.path = path
        self.preproc = Preprocessing()
        self.time_transform = time_transform
        self.start_time = start_time
        self.s_prob = s_prob

    def __iter__(self):
        for line in open(self.path):
            if self.s_prob == 1 or self.s_prob > np.random.rand():
                parts = line.split("\t")
                if self.time_transform != 0 and self.start_time != 0:
                    parts[0] = (float(parts[0]) - self.start_time) / self.time_transform
                try:
                    yield ( float(parts[0]), self.preproc.pp_doc(parts[1]) )
                except:
                    continue


class Preprocessing(object):
    def __init__(self, doc_min_len=6):
        self.stopwords = stopwords.words('english')
        self.doc_min_len = doc_min_len

        self.re_nonan = re.compile(r"[^a-zA-Z0-9_'\- ]")
        self.re_atx = re.compile(r"@\w+")
        self.re_hashtag = re.compile(r"#\w+")
        self.re_wspace = re.compile(r"[ ]+")
        self.re_web = re.compile(r"(?:http:|www)+[\w\-/\.]*")
        self.re_num = re.compile(r"(?:[^a-z]+\d[^a-z]+)")
    
    def pp_doc(self, doc):
        return [x.strip() for x in doc.split() if x not in self.stopwords and len(x) > self.doc_min_len]
    
    def process_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = re.sub(self.re_web, " ", tweet)
        tweet = re.sub(self.re_atx, " ", tweet)
        tweet = re.sub(self.re_num, " ", tweet)
        tags = re.findall(self.re_hashtag, tweet)
        tags = [x.replace("#", "").strip() for x in tags]
        tweet = re.sub(self.re_hashtag, " ", tweet)
        tweet = re.sub(self.re_nonan, " ", tweet)
        tweet = re.sub(self.re_wspace, " ", tweet)
        tweet = " ".join([word.strip() for word in tweet.split() if word not in self.stopwords and len(word) > 2])

        return (tags, tweet.strip())

        
def tweets_to_sentences(in_file, out_file, out_tags_file, offset=4000):
    """Converts SNAP tweets to senteces form and 
    writes everything to out_file. 

    New format is:
    <timestamp>\t<sentence>
    """
    out = open(out_file, 'a')
    out_tags = open(out_tags_file, 'a')
    t_counter = 0
    pp = Preprocessing()
    with open(in_file) as data:
        #Skip first offset lines, usually the beginning has time messed up
        for i in range(offset):
            next(data)
        #Find next blank line
        while True:
            a = next(data)
            if a == "\n":
                break

        t = None
        for line in data:
            if line[0] == "T":
                t = time.mktime(datetime.strptime((" ".join(line.split()[1:])).strip(), "%Y-%m-%d %H:%M:%S").timetuple())
            elif line[0] == "W":
                #Use only tweets with > 10 words
                if len(line.split(" ")) > 10:
                    if t_counter % 100000 == 0:
                        print("{} : {} - Tweets processed".format(in_file, t_counter))
                    t_counter += 1
                    
                    (tags, c_tweet) = pp.process_tweet(" ".join(line.split()[1:]))
                    # Processed must be longer than 8
                    if len(c_tweet.split(" ")) > 8:
                        out.write("{}\t{}\n".format(int(t), c_tweet))
                        
                        if len(tags) > 0:
                            out_tags.write("{}\n".format(" ".join(tags)))
                        else:
                            out_tags.write("{}\n".format("__NONE__"))
    out.close()
    out_tags.close()



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
