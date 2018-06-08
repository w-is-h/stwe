from stwe.utils.time_preprocessing import TimeTransform
from stwe.utils.iterators import TimeWordsIteratorTUW
from stwe.utils.tokenizers import basic_tokenizer
from stwe.utils.loggers import basic_logger

in_files = ['/home/wish/Private/other/stwe/data/stnd_tweets/train/s_tweets.txt']

data_iterator = TimeWordsIteratorTUW(in_files=in_files, tokenizer=basic_tokenizer)

tt = TimeTransform(time_unit='hour', normalize=True)

tt.set_start_end_time(data_iterator=data_iterator)
