import numpy as np

class TokensPreprocessing(object):
    """ This class is an iterator over the dataset, with the
    option to do subsampling of frequent tokens.
    """
    def __init__(self, embedding, data_iterator, subsample=1e-3):
        self.embedding = embedding
        self.data_iterator = data_iterator
        self.vocab = self.embedding.get_vocab()
        self.subsample = subsample

        # Calculate the probability of a word being choosen
        self._calc_word_prob()

    def _calc_word_prob(self):
        vocab = self.vocab

        self.word_freq = [0] * len(vocab)
        self.word_prob = [0] * len(vocab)

        for sample in self.data_iterator:
            tokens = sample[1]

            for token in tokens:
                if token in vocab:
                    self.word_freq[vocab[token].index] += 1

        # Now that we have the count for words we can calculate the 
        #probability of a word being choosen 
        threshold_count = self.subsample * sum(self.word_freq)
        for ind, wfreq in enumerate(self.word_freq):
            if wfreq != 0:
                self.word_prob[ind] = (np.sqrt(wfreq / threshold_count) + 1) * \
                        (threshold_count / wfreq)

    def process_tokens(self, tokens):
        """ This function receives as input a list of tokens, and returns
        a (subsampled) list of indexes. The indexes are taken from the main vocabulary 
        of the embedding object.

        IN: ['i', 'was', 'never', 'a', 'snake']
        OUT: [1, 2, 4]
        """
        out = []
        for token in tokens:
            if token in self.vocab:
                if self.word_prob[self.vocab[token].index] > np.random.rand():
                    out.append(self.vocab[token].index)

        return out

