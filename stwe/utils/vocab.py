import numpy as np

class Vocab(object):
    def __init__(self, data_iterator, embedding, min_count):
        self.v = {}
        self.min_count = min_count
        self.index2token = []
        self.add_words(data_iterator, embedding)


    def as_matrix(self):
        emb_len = len(self.v[self.index2token[0]].vec)
        out = np.zeros((len(self.v), emb_len))

        for i in range(len(self.index2token)):
            w = self.index2token[i]
            out[i] = self.v[w].vec

        return out

    def word_counts(self):
        out = []

        for i in range(len(self.index2token)):
            w = self.index2token[i]
            out.append(self.v[w].count)

        return out




    def add_words(self, data_iterator, embedding):
        ind = 0
        for sample in data_iterator:
            tokens = sample[1]

            for token in tokens:
                if token in self.v:
                    self.v[token].count += 1
                else:
                    if token in embedding:
                        item = VocabItem(
                                vec=embedding[token],
                                cntx_vec=None,#embedding.get_cntx(token),
                                index=ind,
                                count=1)

                        self.v[token] = item
                        ind += 1

        # Remove words with frequency bellow min_count
        for word in list(self.v.keys()):
            if self.v[word].count < self.min_count:
                del self.v[word]

        # Order by frequency, fix indexes and creat index2token
        self.index2token = sorted(list(self.v.keys()), key=lambda k: self.v[k].count, reverse=True)
        for i in range(len(self.index2token)):
            self.v[self.index2token[i]].index = i

    def calc_prob_subsampling(self, subsample):
        """Calculate the probability of a word being choosen when
        subsampling
        """

        treshold_count = sum([x.count for x in self.v.values()]) * subsample
        for word in self.v.keys():
            vi = self.v[word]
            vi.c_prob = (np.sqrt(vi.count / treshold_count) + 1) * \
                    (treshold_count / vi.count)


    def __getitem__(self, word):
        if word in self.v:
            #return unitvec(self.emb.wv.syn0[self.emb.wv.vocab[word].index])
            return self.v[word]
        else:
            return None


    def __contains__(self, word):
        return word in self.v


    def __len__(self):
        return len(self.v)


class VocabItem(object):
    def __init__(self, vec, cntx_vec, index, count, c_prob=1):
        self.vec = vec
        self.cntx_vec = cntx_vec
        self.index = index
        self.count = count
        self.c_prob = c_prob
