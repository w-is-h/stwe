from stwe.utils.iterators import EmbIteratorTUW
from stwe.utils.tokenizers import basic_tokenizer
from stwe.utils.loggers import basic_logger
from embedding.word2vec import Word2Vec
from embedding.fasttext import FastText
from stwe.utils.phraser import Phraser

log = basic_logger('test_word2vec')


#in_files = ['/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_lng/en/test.txt']
#in_files = ['/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_lng/en/tweets.txt']
in_files = ['/home/wish/Private/other/stwe/data/stnd_tweets/raw/test_lng/en/test5m.txt']

data_iterator = EmbIteratorTUW(in_files=in_files, tokenizer=basic_tokenizer)

# Train the Phraser for preprocessing_tokens
phraser = Phraser()
phraser.train(data_iterator)
phraser.save_phrases("/tmp/phraser.dat")

# If Phraser is already trained
#phraser.load_phrases("/tmp/phraser.dat")

# Set the phraser for data_iterator
data_iterator.preprocessing_tokens = phraser.get_phrases

#emb = Word2Vec(save_folder='/tmp/')
emb = FastText(save_folder='/tmp/')

emb.init_train(data_iterator=data_iterator)


# Test
emb.emb.wv.most_similar("open")
