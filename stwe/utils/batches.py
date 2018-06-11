import numpy as np

class BatchGeneratorTUW(object):
    def __init__(self, data_iterator):
        """

        data_iterator:  any iterator that returns (time, ind_tokens),
            the iterator can internally do subsampling or any other
            preprocessing of the tokens/time.
        """
        self.data_iterator = data_iterator

        # Get the lenght of data
        self.data_length = len(self.data_iterator)

    def generate_batch(self, size, max_pairs_from_sample, window_size, max_same_target):
        """ Generates a batch

        size:  the size of the batch
        max_pairs_from_sample:  Given one /tweet/ what is the maximum number of
            input-label pairs to extract, prevents having too much pairs from
            one sample.
        max_same_target:  Given one central word, what is the maximum number
            of words in the context to be choosen. 
        """
        # This calculates the probability of selecting a sample in the data.
        #It is used because batch_size is usally much smaller than the data size.
        select_probability = size / self.data_length / max_pairs_from_sample
        #self.data_iterator.select_probability = select_probability

        # One batch consist of three parts
        # input - being the context /a/ of word /b/
        # labels - being the target word /b/ for which input is the context
        # time - the time where the word pair (a, b) was found
        inputs = []
        labels = []
        times = []

        while len(inputs) < size:
            _break = False
            for sample in self.data_iterator:
                # Number of choosen items from this sample
                nchosen = 0
                # The iterator here must return the /time/ and 
                #an array of ints representing the index of a token
                #in the vocabulary.
                time = sample[0]
                ind_tokens = sample[1]
                l_sample = len(ind_tokens)

                # Randomly select one word:
                for b_ind in np.random.choice(l_sample, l_sample, replace=False):
                    # Now select the second word from the context of the first one
                    _start = max(0, np.random.randint(b_ind - window_size, b_ind))
                    _end = min(l_sample, np.random.randint(b_ind + 1,
                        b_ind + window_size + 2))
                    a_ind_range = np.arange(_start, _end)
                    _nsamples = min(len(a_ind_range), max_same_target)
                    for a_ind in np.random.choice(a_ind_range, _nsamples, replace=False):
                        if a_ind == b_ind:
                            continue
                        inputs.append(ind_tokens[a_ind])
                        labels.append(ind_tokens[b_ind])
                        times.append(time)
                        nchosen += 1

                        if nchosen > max_pairs_from_sample:
                            break
                    if nchosen > max_pairs_from_sample:
                        break
                # Exit the loop over all samples if we have enough
                if len(labels) >= size:
                    break

        # Convert the arrays to numpy and also cut off if longer than size.
        #Needed because of the way the above loops are structured, it can happen 
        #that we get more examples than needed.
        inputs = np.array(inputs[0:size], dtype=int)
        labels = np.array(labels[0:size], dtype=int)
        times = np.array(times[0:size], dtype=np.float32)

        return inputs, labels, times
