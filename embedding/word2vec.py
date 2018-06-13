from .emb import Embedding
from gensim.models import Word2Vec as gWord2Vec
from stwe.utils.loggers import basic_logger

log = basic_logger('embedding_word2vec')

class Word2Vec(Embedding):
    def __init__(self, save_folder, vector_size=300, min_count=30, workers=8, **kwargs):
        super().__init__(save_folder=save_folder,
                        vector_size=vector_size,
                        min_count=min_count)

        # Create the base Word2Vec model from gensim
        self.emb = gWord2Vec(size=vector_size, min_count=min_count, workers=workers, **kwargs)


    def _train(self, data_iterator, epochs, **kwargs):
        total_examples = 0
        for i in data_iterator:
            total_examples += 1
        log.info("There is a total of: {} - examples in the dataset".format(total_examples))
        self.emb.train(data_iterator, total_examples=total_examples, epochs=epochs, **kwargs)


    def init_train(self, data_iterator, pretrained=None, epochs=1, **kwargs):
        self.emb.build_vocab(data_iterator)
        if pretrained is not None:
            log.info("Loading pretrained Vectors in word2vec_format")
            self.emb.intersect_word2vec_format(pretrained, **kwargs)

        self._train(data_iterator, epochs, **kwargs)


    def continue_training(self, data_iterator, epochs=1, **kwargs):
        self.emb.build_vocab(data_iterator, update=True)
        self._train(data_iterator, epochs, **kwargs)


    def get_vocab(self):
        return self.emb.wv.vocab


    def index2word(self, ind):
        return self.emb.wv.index2word[ind]


    def __getitem__(self, word):
        if word in self.emb.wv.vocab:
            #return unitvec(self.emb.wv.syn0[self.emb.wv.vocab[word].index])
            return self.emb.wv.get_vector(word)
        else:
            return None


    def __contains__(self, word):
        return word in self.emb
