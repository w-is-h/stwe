import os

class OptionsSTWE(object):
    def __init__(self):
        # Model options.

        # Where to write out summaries - tensorboard logs.
        self.model_path = "../models/stwe/"
        #self.logs_path = os.path.join(self.model_save_path, "logs")

        # Make the directories if they don't exist
        #os.makedirs(self.model_path, exist_ok=True)
        #os.makedirs(self.logs_path, exist_ok=True)

        # Minimum count for words in the data, all words with frequency bellow
        #this will be removed
        self.min_count = 20

        # Number of clusters to train
        self.nclst = 200

        # Time constant for cluster influence
        self.tau = 1

        # Number of clusters left and right to consider when calculating updates
        self.clst_window = 30

        # Embedding dimension.
        self.emb_dim = 300

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.nepochs = 20

        # The initial learning rate.
        self.learning_rate = 0.0005
        self.min_learning_rate = 0.00001
        self.lr_decay = 1

        # Number of negative samples per example.
        self.nneg = 3

        # Concurrent training steps.
        self.concurrent_steps = 8

        # Number of examples for one training step.
        self.batch_size = 200

        # Number of examples per epoch
        self.epoch_size = 100000

        # The number of words to predict to the left and right of the target word.
        self.window_size = 6

        # Given one sample the maximum number of words (a, b) to take.
        self.max_pairs_from_sample = 200

        # Maximum number of (a, b) pairs from one sample, where the target 'b' is the same word
        self.max_same_target = 10

        # Subsampling threshold for word occurrence.
        self.subsample = 1e-4

        # Seed
        self.seed = 3

        # Beta parametar for v-measure
        self.vm_beta = 1

        # L2 lambda
        self.l2lmbd = 0.000001

        # Files with tags as clusters
        #self.tags_clst_files = None
        # Path to a trained word2vec model
        #self.word2vec_path = "../output/models/word2vec.mdl"

        # The training text file.
        #self.train_data = "../data/train.txt"

        # The testing text file.
        #self.test_data = "../data/test.txt"

        # Timestamp for the first and last document (training data)
        self.start_time = 536454000
        #self._start_time = 536454000
        self.end_time = 1041375600
        #self._end_time = 1041375600
        #self.time_transform = 50000000

