import random

class EmbIteratorTUW(object):
    """ Iterates over the whole dataset, giving only
    sentences/items split into tokens as output. Works with
    input data of tweets in TUW format.
    """

    def __init__(self, in_files, tokenizer, preprocessing_text=None, preprocessing_tokens=None):
        """ Initialization

        in_files:  an array of paths to input files containing tweets in TUW format
        tokenizer:  will be run on each input line to separate the text
        preprocessing_text:  a function that will be run on the text part of TUW
        preprocessing_tokens:  a function that will be run on the tokens got from 
                              the text part of TUW
        """

        self.in_files = in_files
        self.tokenizer = tokenizer
        self.preprocessing_text = preprocessing_text
        self.preprocessing_tokens = preprocessing_tokens


    def __iter__(self):
        for in_file in self.in_files:
            for line in open(in_file):
                # We get the text from a file in TUW format (Time\tUser\tWords)
                text = line.split("\t")[2]
                if self.preprocessing_text is not None:
                    text = self.preprocessing_text(text)

                tokens = self.tokenizer(text)
                if self.preprocessing_tokens is not None:
                    tokens = self.preprocessing_tokens(tokens)

                yield tokens


class TimeWordsIteratorTUW(object):
    """ Iterates over a dataset in TUW format and
    returns the time and tokens as a touple.
    """

    def __init__(self, in_files, tokenizer, preprocessing_time=None, preprocessing_text=None,
            preprocessing_tokens=None, select_probability=1):
        """ Initialization

        in_files:  an array of paths for the input files
        tokenizer:  will be run on each input line to separate the text into tokens
        preprocessing_time:  Function that will be run on the T part of TUW
        preprocessing_text:  Function that will be run on the W part of TUW
        preprocessing_tokens:  a function that will be run on the tokens got from
                              the text part of TUW
        select_probability:  The probability of selecting a line in the
            input files. Used for input subsampling.

        yield:  (time, tokens)
        """
        self.in_files = in_files
        self.tokenizer = tokenizer
        self.preprocessing_time = preprocessing_time
        self.preprocessing_text = preprocessing_text
        self.select_probability = select_probability
        self.preprocessing_tokens = preprocessing_tokens
        self._length = None

    def __iter__(self):
        for in_file in self.in_files:
            for line in open(in_file):
                if self.select_probability == 1 or random.random() < self.select_probability:
                    # We get the text/time from a file in TUW format (Time\tUser\tWords)
                    tmp = line.split("\t")
                    time = float(tmp[0])
                    text = tmp[2]

                    if self.preprocessing_text is not None:
                        text = self.preprocessing_text(text)

                    if self.preprocessing_time is not None:
                        time = self.preprocessing_time(time)

                    tokens = self.tokenizer(text)
                    if self.preprocessing_tokens is not None:
                        tokens = self.preprocessing_tokens(tokens)

                    yield (time, tokens)

    def __len__(self):
        if self._length is None:
            i = 0
            old_s_p = self.select_probability
            self.select_probability = 1
            for x in self:
                i += 1
            self._length = i
            self.select_probability = old_s_p
        return self._length
