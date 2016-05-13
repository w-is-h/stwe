import numpy as np
import operator
import tensorflow as tf
from gensim.models import Word2Vec
from io_utils import DocumentTimePair

"""
flags = tf.app.flags
flags.DEFINE_string("save_path", "../output/model/", "Directory to write the model.")
flags.DEFINE_string("word2vec_path", "../output/models/word2vec.dat", "Path to trained word2vec model")
flags.DEFINE_string("train_data", None, "Training data.")
flags.DEFINE_string("test_data", None, "Testing data")
flags.DEFINE_integer("start_time", None, "Timestamp of the first document")
flags.DEFINE_integer("end_time", None, "Timestamp of the last document")
flags.DEFINE_integer("nclst", None, "Number of clusters")
flags.DEFINE_float("tau", None, "Time constant for cluster influence")
flags.DEFINE_integer("clst_window", 50, "Number of clusters around current one"
                        " to consider when calculating updates")
flags.DEFINE_float("train_subsampling", 0.0001, "Probability of taking one example"
                        "in the train dataset")
flags.DEFINE_integer("emb_dim", 300, "The embedding dimension size.")
flags.DEFINE_integer("nepochs", 15, "Number of epochs to train. " 
                        "Each epoch processes the training data once "
                        "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("nneg", 25, "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500, "Numbers of training examples each step processes ")
flags.DEFINE_integer("concurrent_steps", 12, "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                        "The number of words to predict to the left and right "
                        "of the target word.")
flags.DEFINE_integer("max_pairs_from_sample", 30,
                        "Given one sample them maximum number of words (a, b) to take.")
flags.DEFINE_integer("max_same_target", 30,
                        "Maximum number of (a, b) pairs from one sample, where the target 'b'"
                        " is the same word")
flags.DEFINE_float("subsample", 1e-3,
                        "Subsample threshold for word occurrence. Words that appear "
                        "with higher frequency will be randomly down-sampled. Set "
                        "to 0 to disable.")
flags.DEFINE_integer("seed", 3, "Seed")
flags.DEFINE_float("vm_beta", 1.2, "Beta parametar for v-measure")
FLAGS = flags.FLAGS
"""

class Options(object):
    """Options used by our word2vec model."""
    def __init__(self):
        # Model options.

        # Where to write out summaries.
        self.save_path = "../output/model/"

        # Path to a trained word2vec model
        self.word2vec_path = "../output/models/word2vec.dat"

        # The training text file.
        self.train_data = "../data/final.txt"

        # The testing text file.
        self.test_data = ""

        # Timestamp for the first and last document (training data)
        self.start_time = 536454000
        self.end_time = 1041375600

        # Number of clusters to train
        self.nclst = 200

        # Time constant for cluster influence
        self.tau = 1e-7

        # Number of clusters left and right to consider when calculating updates
        self.clst_window = 50

        # Embedding dimension.
        self.emb_dim = 100

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.nepochs = 100

        # The initial learning rate.
        self.learning_rate = 0.001

        # Number of negative samples per example.
        self.nneg = 2

        # Concurrent training steps.
        self.concurrent_steps = 6

        # Number of examples for one training step.
        self.batch_size = 100
        
        self.epoch_size = 100000

        # The number of words to predict to the left and right of the target word.
        self.window_size = 20

        # Given one sample the maximum number of words (a, b) to take. 
        self.max_pairs_from_sample = 100

        # Maximum number of (a, b) pairs from one sample, where the target 'b' is the same word
        self.max_same_target = 10

        # Subsampling threshold for word occurrence.
        self.subsample = 1e-3

        # Seed
        self.seed = 3

        # Beta parametar for v-measure
        self.vm_veta = 1

        # Top clusters to use for updating

    """
    def __init__(self):
        # Model options.

        # Where to write out summaries.
        self.save_path = FLAGS.save_path

        # Path to a trained word2vec model
        self.word2vec_path = FLAGS.word2vec_path

        # The training text file.
        self.train_data = FLAGS.train_data

        # The testing text file.
        self.test_data = FLAGS.test_data

        # Timestamp for the first and last document (training data)
        self.start_time = FLAGS.start_time
        self.end_time = FLAGS.end_time

        # Number of clusters to train
        self.nclst = FLAGS.nclst

        # Time constant for cluster influence
        self.tau = FLAGS.tau

        # Number of clusters left and right to consider when calculating updates
        self.clst_window = FLAGS.clst_window

        # Probability of taking one example (training data)
        self.train_subsampling = FLAGS.train_subsampling

        # Embedding dimension.
        self.emb_dim = FLAGS.emb_dim

        # Number of epochs to train. After these many epochs, the learning
        # rate decays linearly to zero and the training stops.
        self.nepochs = FLAGS.nepochs

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of negative samples per example.
        self.nneg = FLAGS.nneg

        # Concurrent training steps.
        self.concurrent_steps = FLAGS.concurrent_steps

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # The number of words to predict to the left and right of the target word.
        self.window_size = FLAGS.window_size

        # Given one sample them maximum number of words (a, b) to take. 
        self.max_pairs_from_sample = FLAGS.max_pairs_from_sample

        # Maximum number of (a, b) pairs from one sample, where the target 'b' is the same word
        self.max_same_target = FLAGS.max_same_target

        # Subsampling threshold for word occurrence.
        self.subsample = FLAGS.subsample

        # Seed
        self.seed = FLAGS.seed

        # Beta parametar for v-measure
        self.vm_veta = FLAGS.vm_beta
    """

class TempWordEmb(object):
    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._word2id = {}
        self._id2word = []
        # Central and Context representations from word2vec
        self._cent_word = []
        self._cent_cntx = []

        self.build_graph()

    def build_graph(self):
        opts = self._options

        # Load the data: vocab, cent_word, cent_cntx
        self.load_data()
        print("Vocabulary size: ", opts.vocab_size)

        # Make an init batch
        _, _, _ = self.generate_batch(opts.train_data)

        # Build graph for the forward pass
        true_logits, sampled_logits = self.forward()

        true_logits, sampled_logits = self.forward()
        loss = self.nce_loss(true_logits, sampled_logits)
        tf.scalar_summary("NCE loss", loss)

        self._loss = loss
        self.optimize(loss)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()


    def load_data(self):
        opts = self._options

        # Load word2vec model
        word2vec = Word2Vec.load(opts.word2vec_path)

        # Sort the vocabulary from word2vec, descending - most frequent word first
        sorted_vocab_pairs = sorted(word2vec.vocab.items(), key=operator.itemgetter(1), reverse=True)

        # Fill id2word array
        for pair in sorted_vocab_pairs:
            self._id2word.append(pair[0])

        # Fill word2id dictionary
        for ind, value in enumerate(self._id2word):
            self._word2id[value] = ind

        # Fill _cent_word and _cent_cntx
        for word in self._id2word:
            self._cent_word.append(word2vec.syn0[word2vec.vocab[word].index])
            self._cent_cntx.append(word2vec.syn1neg[word2vec.vocab[word].index])

        opts.vocab_size = len(self._id2word)


    def forward(self):
        opts = self._options

        # Rho - the probability of a word appearing in a cluster - dim #words x #clusters
        rho = tf.Variable(tf.zeros([opts.vocab_size, opts.nclst]), name='rho')

        # word_clst - Clusters for embedded words
        word_clst = tf.Variable(tf.random_uniform([opts.nclst, opts.emb_dim], -0.5, 0.5), name='word_clst')
        # cntx_clst - Clusters for context
        cntx_clst = tf.Variable(tf.random_uniform([opts.nclst, opts.emb_dim], -0.5, 0.5), name='cntx_clst')

        # clst_time - Timestamp for each cluster
        clst_time = tf.Variable(tf.random_uniform([opts.nclst, 1], opts.start_time, opts.end_time, 
            dtype=tf.int32), name='clst_time')

        # Training data - [index_for_vector, time]
        self.train_inputs = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='train_inputs')
        train_inputs = self.train_inputs
        # Labels for the training data
        self.train_labels = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='train_labels')
        train_labels = self.train_labels
        # Time for every pair (a, b)
        self.train_inputs_time = tf.placeholder(tf.int32, shape=[opts.batch_size, 1])
        train_inputs_time = self.train_inputs_time

        # cent_word - Central representation for word representations
        self.cent_word = tf.placeholder(tf.float32, shape=[opts.vocab_size, opts.emb_dim], name='cent_word')
        cent_word = self.cent_word
        word_emb = tf.nn.embedding_lookup(cent_word, tf.reshape(train_labels, [-1]))
        # cent_cntx - Central representation for context representations
        self.cent_cntx = tf.placeholder(tf.float32, shape=[opts.vocab_size, opts.emb_dim], name='cent_cntx')
        cent_cntx = self.cent_cntx
        cntx_emb = tf.nn.embedding_lookup(cent_cntx, tf.reshape(train_inputs, [-1]))

        # Rho lookup
        rho_lup_cntx = tf.nn.embedding_lookup(rho, tf.reshape(train_inputs, [-1]))

        # Time difference between word and cluster time
        t_ctime = tf.tile(clst_time, [opts.batch_size, 1])
        # Time for every word repated for number of clusters and flatened 
        t_wtime = tf.reshape(tf.tile(train_inputs_time, [1, opts.nclst]), [-1 ,1])

        time_diff = -1 * tf.reshape(tf.abs(t_ctime - t_wtime), [-1, opts.nclst])
        selected_time_diff, selected_time_diff_ind = tf.nn.top_k(time_diff, k=opts.clst_window)

        flat_selected_time_diff_ind = tf.reshape(selected_time_diff_ind + tf.reshape(
            tf.range(0, opts.batch_size) * opts.nclst, [-1, 1]), [-1])

        # We have to flatten the rho_lup to match the flat_selected_time_diff_ind
        flat_rho_lup_cntx = tf.reshape(rho_lup_cntx, [-1, 1])

        # Rhos for selected clusters based on time difference between word and clst_time
        selected_rho_cntx = tf.gather(flat_rho_lup_cntx, flat_selected_time_diff_ind)

        # Make t vars for calculating the transform function g_c(x_a)
        #selected_clst are cluster embeddings for every context in the input
        #t_a is all words repated M times, where M is the number of clusters used
        selected_clst_cntx = tf.nn.embedding_lookup(cntx_clst, tf.reshape(selected_time_diff_ind, [-1])) 
        t_a_cntx = tf.reshape(tf.tile(train_inputs, [1, opts.clst_window]), [-1])
        selected_emb_cntx = tf.nn.embedding_lookup(cent_cntx, t_a_cntx)

        fc = tf.exp(tf.to_float( tf.reshape(selected_time_diff, [-1, 1]) ) * opts.tau)
        gc_cntx = selected_clst_cntx - selected_emb_cntx

        dynamic_repr_cntx = tf.reduce_sum( tf.reshape(gc_cntx * selected_rho_cntx * fc, 
            [opts.batch_size, opts.emb_dim, opts.clst_window]), 2) 

        # The final representation (static + dynamic) of all contexts in the input dataset
        all_repr_cntx = cntx_emb + dynamic_repr_cntx

        ### Calculation of the word (label) representation
        rho_lup_word =  tf.nn.embedding_lookup(rho, tf.reshape(train_labels, [-1]))
        # We have to flatten the rho_lup to match the flat_selected_time_diff_ind
        flat_rho_lup_word = tf.reshape(rho_lup_word, [-1, 1])

        # Rhos for selected clusters based on time difference between word and clst_time
        selected_rho_word = tf.gather(flat_rho_lup_cntx, flat_selected_time_diff_ind)

        # Make t vars for calculating the transform function g_c(x_a)
        #selected_clst are cluster embeddings for every context in the input
        #t_a is all words repated M times, where M is the number of clusters used
        selected_clst_word = tf.nn.embedding_lookup(word_clst, tf.reshape(selected_time_diff_ind, [-1]))
        t_a_word = tf.reshape(tf.tile(train_labels, [1, opts.clst_window]), [-1])
        selected_emb_word = tf.nn.embedding_lookup(cent_word, t_a_word)

        gc_word = selected_clst_word - selected_emb_word

        dynamic_repr_word = tf.reduce_sum( tf.reshape(gc_word * selected_rho_word * fc,
            [opts.batch_size, opts.emb_dim, opts.clst_window]), 2)

        # The final representation (static + dynamic) of all contexts in the input dataset
        all_repr_word = word_emb + dynamic_repr_word


        ### Negative sampling
        labels_matrix = tf.reshape(tf.cast(train_labels, dtype=tf.int64), [opts.batch_size, 1])

        negative_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix,
            num_true=1,
            num_sampled=opts.batch_size,
            unique=False,
            range_max=opts.vocab_size,
            distortion=0.75,
            unigrams=opts.train_data_params['cnts'].tolist()))

        # Bring negative ids to correct shape
        negative_ids = tf.reshape(negative_ids, [-1, 1])

        # Rho lookup
        rho_lup_neg = tf.nn.embedding_lookup(rho, tf.reshape(negative_ids, [-1]))

        # We have to flatten the rho_lup to match the flat_selected_time_diff_ind
        flat_rho_lup_neg = tf.reshape(rho_lup_neg, [-1, 1])

        # Rhos for selected clusters based on time difference between word and clst_time
        selected_rho_neg = tf.gather(flat_rho_lup_neg, flat_selected_time_diff_ind)

        # Make t vars for calculating the transform function g_c(x_a)
        #selected_clst are cluster embeddings for every context in the input
        #t_a is all words repated M times, where M is the number of clusters used
        selected_clst_neg = tf.nn.embedding_lookup(cntx_clst, tf.reshape(selected_time_diff_ind, [-1])) 
        t_a_neg = tf.reshape(tf.tile(negative_ids, [1, opts.clst_window]), [-1])
        selected_emb_neg = tf.nn.embedding_lookup(cent_cntx, t_a_neg)

        gc_neg = selected_clst_neg - selected_emb_neg

        dynamic_repr_neg = tf.reduce_sum( tf.reshape(gc_neg * selected_rho_neg * fc, 
            [opts.batch_size, opts.emb_dim, opts.clst_window]), 2) 

        # The final representation (static + dynamic) of all contexts in the input dataset
        neg_emb = tf.nn.embedding_lookup(cent_cntx, tf.reshape(negative_ids, [-1]))
        all_repr_neg = neg_emb + dynamic_repr_neg

        # A hack allowing us to have more than one negative word per example.
        t_all_repr_neg = tf.tile(all_repr_neg, [opts.nneg, 1])
        t_all_repr_word = tf.reshape(tf.tile(all_repr_word, [1, opts.nneg]), [opts.nneg * opts.batch_size, -1])

        t_negative_logits = tf.reduce_sum(tf.mul(t_all_repr_word, t_all_repr_neg), 1) #+ true_b
        negative_logits = tf.reshape(t_negative_logits, [-1, opts.nneg])

        true_logits = tf.reduce_sum(tf.mul(all_repr_word, all_repr_cntx), 1) #+ true_b

        return true_logits, negative_logits

    def generate_fullbatch(self, data, dataset_name='test'):
        data = DocumentTimePair(data)

        # Generate a batch using all samples and all words from a sample
        opts = self._options
        data_params = {}

        # Check do we already have the data length
        if hasattr(opts, dataset_name + '_data_params'):
            data_params = getattr(opts, dataset_name + '_data_params')
        else:
            data_params = self.get_data_params(data)
            setattr(opts, dataset_name + '_data_params', data_params)

        batch = []
        labels = []
        time = []

        for example in data:
            t = example[0]
            tmp = example[1]
            # Filter words in doc by probability 
            #TODO: None or nothing
            doc = [w if w in self._word2id and (data_params['probs'][self._word2id[w]] >= 1 or 
                data_params['probs'][self._word2id[w]] > np.random.rand()) else None for w in tmp]
            for wid in np.random.choice(len(doc), len(doc), replace=False):
                # Choose second word from the window around the first word
                if doc[wid] is None:
                    continue
                for wid2 in np.arange(max(0,   np.random.randint(wid - opts.window_size, wid)),
                        min(len(doc), np.random.randint(wid + 1, wid + opts.window_size + 2))):
                    if doc[wid2] == doc[wid] or doc[wid2] is None:
                        continue
                    batch.append(self._word2id[doc[wid]])
                    labels.append(self._word2id[doc[wid2]])
                    time.append(t)

        return batch, labels, time

    def generate_batch(self, data, dataset_name='train'):
        data = DocumentTimePair(data)

        opts = self._options
        data_params = {}

        # Check do we already have the data length
        if hasattr(opts, dataset_name + '_data_params'):
            data_params = getattr(opts, dataset_name + '_data_params')
        else:
            data_params = self.get_data_params(data)
            setattr(opts, dataset_name + '_data_params', data_params)

        # The probability of choosing one item to be part of the batch.
        #It has the effect of random choice of a certain size from the
        #whole dataset.
        select_prob = opts.epoch_size / data_params['len'] / opts.max_pairs_from_sample

        batch = np.zeros((opts.epoch_size, 1), dtype=int)
        labels = np.zeros((opts.epoch_size, 1), dtype=int)
        time = np.zeros((opts.epoch_size, 1), dtype=int)
        n_added = 0
        test = []
        while True:
            for example in data:
                if select_prob > np.random.rand():
                    t = example[0]
                    tmp = example[1]
                    doc = []
                    #Filter words in doc by probability 
                    #TODO: None or nothing
                    doc = [w if w in self._word2id and (data_params['probs'][self._word2id[w]] >= 1 or 
                        data_params['probs'][self._word2id[w]] > np.random.rand()) else None for w in tmp]
                    added_pairs = 0
                    # Randomly choose first word
                    if len(doc) > 0:
                        for wid in np.random.choice(len(doc), len(doc), replace=False):
                            # Choose second word from the window around the first word
                            if doc[wid] is None:
                                continue

                            range_wid2 = np.arange(max(0,   np.random.randint(wid - opts.window_size, wid)),
                                    min(len(doc), np.random.randint(wid + 1, wid + opts.window_size + 2)))
                            # Select random words from the window of 'wid', limit number to max_same_target
                            for wid2 in np.random.choice(range_wid2, min(len(range_wid2), opts.max_same_target)):
                                # Skip a pair containing the same words, or if the second word is not 
                                #in the vocab
                                if doc[wid2] == doc[wid] or doc[wid2] is None:
                                    continue

                                batch[n_added] = self._word2id[doc[wid]]
                                labels[n_added] = self._word2id[doc[wid2]]
                                time[n_added] = t

                                n_added += 1
                                added_pairs += 1

                                if added_pairs == opts.max_pairs_from_sample or n_added >= opts.epoch_size:
                                    break
                            if added_pairs == opts.max_pairs_from_sample or n_added >= opts.epoch_size:
                                break
                if n_added >= opts.epoch_size:
                    break
            if n_added >= opts.epoch_size:
                break
        return batch, labels, time

    def get_data_params(self, data):
        opts = self._options
        i = 0
        cnts = np.zeros(len(self._word2id), dtype=int)
        probs = np.zeros(len(self._word2id), dtype=float)

        for one in data:
            i += 1
            for word in one[1]:
                if word in self._word2id:
                    cnts[self._word2id[word]] += 1


        opts.subsample = 1e-3
        total_words = np.sum(cnts)
        threshold_count = float(opts.subsample) * total_words
        for ind, cnt in enumerate(cnts):
            if cnt > 0:
                probs[ind] = (np.sqrt(cnt / float(threshold_count)) + 1) * \
                        (threshold_count / float(cnt)) if opts.subsample else 1

        return {"len": i, "cnts": cnts, "probs": probs}


    def train(self, epoch):
        opts = self._options

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(opts.save_path, self._session.graph)
        workers = []

        batch, labels, time = self.generate_batch(opts.train_data)
        for i in range(opts.epoch_size // opts.batch_size):
            feed_dict={self.train_inputs: batch[i*opts.batch_size:(i+1)*opts.batch_size], 
                self.train_labels: labels[i*opts.batch_size:(i+1)*opts.batch_size],
                self.train_inputs_time: time[i*opts.batch_size:(i+1)*opts.batch_size],
                self.cent_word: self._cent_word,
                self.cent_cntx: self._cent_cntx}

            self._session.run([self._train], feed_dict)
            if (i * opts.batch_size) % 10000 == 0:
                summary_str = self._session.run(summary_op, feed_dict)
                summary_writer.add_summary(summary_str, epoch * opts.epoch_size + i*opts.batch_size)
        

    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        opts = self._options

        # Optimizer nodes.
        # Linear learning rate decay.
        optimizer = tf.train.AdamOptimizer(opts.learning_rate, epsilon=1e-4)
        train = optimizer.minimize(loss, gate_gradients=optimizer.GATE_NONE)
        self._train = train

    def nce_loss(self, true_logits, negative_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        opts = self._options
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                true_logits, tf.ones_like(true_logits))
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                negative_logits, tf.zeros_like(negative_logits))

        # NCE-loss is the sum of the true and noise (negative words)
        # contributions, averaged over the batch.
        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                tf.reduce_sum(negative_xent)) / opts.batch_size
        return nce_loss_tensor


def main(_):
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            twe = TempWordEmb(opts, session)
            for i in range(opts.nepochs):
                print("Start epoch: {}".format(i))
                twe.train(i)
            print("fdsfsdf")


