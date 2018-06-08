def BatchGeneratorTUW(object):
    def __init__(self, data_iterator):
        """

        data_iterator:  any iterator that returns (time, tokens)
        """
        self.data_iterator = data_iterator

        # Get the lenght of data
        self.data_length = len(self.data_iterator)

    def generate_batch(self, size, max_pairs_from_sample):
        # This calculates the probability of selecting a sample in the data.
        #It is used because batch_size is usally much smaller than the data size.
        select_probability = size / self.data_length / max_pairs_from_sample

        # On batch consist of three parts
        # input - being the context /a/ of word /b/
        # labels - being the target word /b/ for which input is the context
        # time - the time where the word pair (a, b) was found
        inputs = np.zeros((size, 1), dtype=int)
        labels = np.zeros((size, 1), dtype=int)
        times = np.zeros((size, 1), dtype=int)

        while True:
            for sample in self.data_iterator:
                time = sample[0]
                tokens = sample[1]

