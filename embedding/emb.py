import os
from os.path import join as p_join
import pickle
from stwe.utils.loggers import basic_logger

log = basic_logger('embedding_word2vec')


class Embedding(object):
    name = "embedding"

    def __init__(self, save_folder, vector_size=300,  min_count=11):
        # Make the directories in the path if they don't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.save_folder = save_folder
        self.vector_size = vector_size
        self.min_count = min_count
        self.emb = None


    def init_train(self, data_iterator, pretrained=None, binary=False):
        print("NOT IMPLEMENTED")
        pass


    def continue_training(self, data_iterator):
        print("NOT IMPLEMENTED")
        pass


    def get_vocab(self):
        print("NOT IMPLEMENTED")
        pass


    def index2word(self, word):
        print("NOT IMPLEMENTED")
        pass


    def __getitem__(self, word):
        print("NOT IMPLEMENTED")
        pass


    def __contains__(self, word):
        print("NOT IMPLEMENTED")
        pass


    def save(self, save_file=None):
        f = open(p_join(self.save_folder, self.name + ".dat"), 'wb')
        pickle.dump(self, f)
        f.close()

        log.info("File saved in: " + f.name)


    @classmethod
    def load(self, save_folder):
        f = open(p_join(save_folder, self.name + ".dat"), 'rb')

        return pickle.load(f)
