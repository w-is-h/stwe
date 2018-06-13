import numpy as np

class TokensPreprocessing(object):
    """ This class is an iterator over the dataset, with the
    option to do subsampling of frequent tokens.
    """
    def __init__(self, vocab):
        self.vocab = vocab


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
                if self.vocab[token].c_prob > np.random.rand():
                    out.append(self.vocab[token].index)

        return out
