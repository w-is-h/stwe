from gensim.models import phrases as gensim_phrases

class Phraser:
    def __init__(self):
        self.phraser = None

    def train(self, data_iterator, **kwargs):
        # Train the phraser from gensim
        self.phraser = gensim_phrases.Phraser(gensim_phrases.Phrases(data_iterator, **kwargs))

    def save_phrases(self, filename):
        self.phraser.save(filename)

    def load_phrases(self, filename):
        self.phraser = gensim_phrases.Phraser.load(filename)

    def get_phrases(self, tokens):
        return self.phraser[tokens]
