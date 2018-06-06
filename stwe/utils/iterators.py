class EmbIteratorTUW(object):
    """ Iterates over the whole dataset, giving only
    sentences/items split into tokens as output. Works with
    input data of tweets in TUW format.
    """

    def __init__(self, in_files, tokenizer, preprocessing_text=None, preprocessing_tokens=None):
        """ Initialization

        in_files: an array of paths to input files containing tweets in TUW format
        tokenizer: will be run on each input line to separate the text
        preprocessing_text: a function that will be run on the text part of TUW
        preprocessing_tokens: a function that will be run on the tokens got from 
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
