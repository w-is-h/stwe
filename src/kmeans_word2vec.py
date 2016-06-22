import numpy as np
from io_utils import read_cent_repr, DocumentTimePair
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import pickle
from gensim import matutils

class KmeansWord2Vec():
    def __init__(self, nclst, repr_file):
        self.nclst = nclst
        self.repr = self.read_repr(repr_file)

    def read_repr(self, in_file):
        return read_cent_repr(in_file)


    def _process_tweet(self, tweet):
        data = None
        n_words = 0
        for word in tweet:
            if word in self.repr:
                n_words += 1
                if data is None:
                    data = self.repr[word]
                else:
                    data = data + self.repr[word]
        if data is not None:
            return data / n_words
        else:
            return np.zeros(100)

    def calc_kmeans(self, tweets_file, niter=100, random_range=(0, 10)):
        mb_kmeans = MiniBatchKMeans(self.nclst, batch_size=1)
        t_stream = DocumentTimePair(tweets_file)        
        for iter in range(niter):
            #Create dataset
            data = []
            random_skip = np.random.randint(random_range[0], random_range[1])
            cnt = 0
            for tweet in t_stream:
                tweet = tweet[1]
                if cnt < random_skip:
                    cnt = cnt + 1
                    continue
                random_skip = np.random.randint(random_range[0], random_range[1])
                cnt = 0
                data.append(self._process_tweet(tweet))
            print("Kmeans iteration: {} out of {} with dataset of: {}".format(iter, niter, len(data)))
            data = np.array(data)
            mb_kmeans.partial_fit(data)

        self.kmeans = mb_kmeans


    def predict(self, tweets_file, out_file, batch_size=100000):
        t_stream = DocumentTimePair(tweets_file)        
        cnt = 0
        out = open(out_file, 'w')
        data = []
        for tweet in t_stream:
            tweet = tweet[1]
            data.append(self._process_tweet(tweet))
            cnt += 1
            if cnt == batch_size:
                data = np.array(data)
                pred = self.kmeans.predict(data)
                for one in pred:
                    out.write("%d\n" % one)
                cnt = 0
                data = []
        if cnt != 0:
            data = np.array(data)
            pred = self.kmeans.predict(data)
            for one in pred:
                out.write("%d\n" % one)

        out.close() 


    def kmeans_words(self, b_dict_file, means_file):
        w_repr = np.array(list(self.repr.values()), dtype=float)
        kmeans = KMeans(self.nclst)
        kmeans = kmeans.fit(w_repr)
        means = kmeans.cluster_centers_

        f = open(means_file, "wb")
        pickle.dump(means, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        for ind in range(means.shape[0]):
            means[ind] = matutils.unitvec(means[ind])
        means = means.transpose()
        b_dict = {}
        for key in self.repr.keys():
            b_dict[key] = np.dot(matutils.unitvec(self.repr[key]), means)

        f = open(b_dict_file, "wb")
        pickle.dump(b_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()

