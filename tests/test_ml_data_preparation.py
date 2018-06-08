from embedding.word2vec import Word2Vec
from stwe.utils.tokens_preprocessing import TokensPreprocessing
from stwe.utils.time_preprocessing import TimeTransform
from stwe.utils.iterators import TimeWordsIteratorTUW
from stwe.utils.tokenizers import basic_tokenizer
from stwe.utils.loggers import basic_logger

in_files = ['/home/wish/Private/other/stwe/data/stnd_tweets/train/s_tweets.txt']

emb = Word2Vec.load(save_folder='/home/wish/Private/other/stwe/models/emb/w2v/')

data_iterator = TimeWordsIteratorTUW(in_files=in_files, tokenizer=basic_tokenizer)

# Initialize the time transform class
tt = TimeTransform(time_unit='hour', normalize=True)
# Find the start and end time of the data
tt.set_start_end_time(data_iterator=data_iterator)

# Set the time transform function for the data_iterator
data_iterator.preprocessing_time = tt.transform

tokens_preprocessing = TokensPreprocessing(emb, data_iterator)
