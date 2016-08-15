import numpy as np
import gensim
import vmeasure
import operator
import tensorflow as tf
from gensim.models import Word2Vec
from io_utils import DocumentTimePair
from queue import Queue
from threading import Thread

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
        self.word2vec_path = "../output/models/word2vec.mdl"

        # The training text file.
        self.train_data = "../data/train.txt"

        # The testing text file.
        self.test_data = "../data/test.txt"

        self.min_count = 50

        # Timestamp for the first and last document (training data)
        self.start_time = 536454000
        self._start_time = 536454000
        self.end_time = 1041375600
        self._end_time = 1041375600
        self.time_transform = 50000000

        # Number of clusters to train
        self.nclst = 200

        # Time constant for cluster influence
        self.tau = 1

        # Number of clusters left and right to consider when calculating updates
        self.clst_window = 30

        # Embedding dimension.
        self.emb_dim = 100

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
        self.vm_beta = 1

        # L2 lambda
        self.l2lmbd = 0.000001

        # Files with tags as clusters
        self.tags_clst_files = None

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
        # Used so that the testset is keept in memory
        self._testset = None
        self._options = options
        self._epoch = 0
        
        if self._options.time_transform > 0:
            self._options.end_time = (self._options.end_time - self._options.start_time) / self._options.time_transform
            self._options.start_time = 0

        self._session = session
        self._word2id = {}
        self._id2word = []
        # Central and Context representations from word2vec
        self._cent_word = []
        self._cent_cntx = []

        self._train_loss = 0
        self._test_loss = 0

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

        self._loss = loss
        
        
        #self.optimize(loss)
        #Aleterations
        self.optimize_a2(loss)
        self.optimize_a1(loss)

        # Variables for train and test loss
        self.train_loss = tf.placeholder(tf.float32)
        self.test_loss = tf.placeholder(tf.float32)
        tf.scalar_summary("train_loss", self.train_loss)
        tf.scalar_summary("test_loss", self.test_loss)

        for i in range(opts.nclst):
            tf.scalar_summary("clsttime_{}".format(i), tf.reshape(self.clst_time, [-1])[i])

        # Histogram for time
        tf.histogram_summary("clst_time", self.clst_time)
        # Histogram for tau
        tf.histogram_summary("tau", self.tau)

        # Summaries - Rho 
        tf.scalar_summary("rho_min", tf.reduce_min(self.rho))
        tf.scalar_summary("rho_max", tf.reduce_max(self.rho))
        tf.scalar_summary("rho_avg", tf.reduce_mean(self.rho))

        # Summaries - cntx_clst
        cntx_clst_sq = tf.square(self.cntx_clst)
        enorm_cntx_clst = tf.sqrt(tf.reduce_sum(cntx_clst_sq, 1))
        tf.scalar_summary("cntxclst_min", tf.reduce_min(enorm_cntx_clst))
        tf.scalar_summary("cntxclst_max", tf.reduce_max(enorm_cntx_clst))
        tf.scalar_summary("cntxclst_avg", tf.reduce_mean(enorm_cntx_clst))

        # Summaries - word_clst
        word_clst_sq = tf.square(self.word_clst)
        enorm_word_clst = tf.sqrt(tf.reduce_sum(word_clst_sq, 1))
        tf.scalar_summary("wordclst_min", tf.reduce_min(enorm_word_clst))
        tf.scalar_summary("wordclst_max", tf.reduce_max(enorm_word_clst))
        tf.scalar_summary("wordclst_avg", tf.reduce_mean(enorm_word_clst))

        # Time prediction

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()


    def load_data(self):
        opts = self._options

        # Load word2vec model
        word2vec = Word2Vec.load(opts.word2vec_path)

        # Sort the vocabulary from word2vec, descending - most frequent word first
        sorted_vocab_pairs = sorted(word2vec.vocab.items(), key=operator.itemgetter(1), reverse=True)

        vocab_from_data = {}
        dtp = DocumentTimePair(opts.train_data) 
        for pair in dtp:
            snt = pair[1]
            for word in snt:
                if word in vocab_from_data:
                    vocab_from_data[word] += 1
                else:
                    vocab_from_data[word] = 1
        
        #Remove pairs with frequency lower than min
        tmp = {}
        for key in vocab_from_data.keys():
            if vocab_from_data[key] > opts.min_count:
                tmp[key] = vocab_from_data[key]
        vocab_from_data = tmp


        # Fill id2word array
        for pair in sorted_vocab_pairs:
            if pair[0] in vocab_from_data:
                self._id2word.append(pair[0])

        # Fill word2id dictionary
        for ind, value in enumerate(self._id2word):
            self._word2id[value] = ind

        # Fill _cent_word and _cent_cntx
        for word in self._id2word:
            #Normalization
            #self._cent_word.append(gensim.matutils.unitvec(word2vec.syn0[word2vec.vocab[word].index]) / 100.0)
            #self._cent_cntx.append(gensim.matutils.unitvec(word2vec.syn1neg[word2vec.vocab[word].index]) / 100.0)
            
            self._cent_word.append(word2vec.syn0[word2vec.vocab[word].index])
            self._cent_cntx.append(word2vec.syn1neg[word2vec.vocab[word].index])


        opts.vocab_size = len(self._id2word)
    
    def pred_time(self):
        opts = self._options

        self.eval_doc = tf.placeholder(tf.int32, shape=[None], name='eval_doc')
        self.eval_doc_time = tf.placeholder(tf.float32, name='eval_doc_time')

        doc_rho = tf.nn.embedding_lookup(self.rho, self.eval_doc)

        doc_probs = tf.reduce_prod(tf.sigmoid(doc_rho), 0)

        max_prob_ind = tf.argmax(doc_probs, 0)

        predicted_time = tf.gather(self.clst_time, max_prob_ind)

        self.pred_time_error = tf.reduce_mean(tf.abs(self.eval_doc_time - predicted_time))


    def forward(self):
        opts = self._options

        self.epoch = tf.placeholder(tf.int32)

        # Rho - the probability of a word appearing in a cluster - dim #words x #clusters
        self.rho = tf.Variable(tf.ones([opts.vocab_size, opts.nclst]), name='rho') * -5

        # Tau - the time constraint on cluster influence
        self.tau = tf.Variable(tf.ones([opts.nclst, 1]), name='tau') * opts.tau

        # word_clst - Clusters for embedded words
        self.word_clst = tf.Variable(tf.random_uniform([opts.nclst, opts.emb_dim], -0.5, 0.5), name='word_clst')
        # cntx_clst - Clusters for context
        self.cntx_clst = tf.Variable(tf.random_uniform([opts.nclst, opts.emb_dim], -0.5, 0.5), name='cntx_clst')

        # clst_time - Timestamp for each cluster
        self.clst_time = tf.Variable(tf.random_uniform([opts.nclst, 1], opts.start_time, opts.end_time, 
            dtype=tf.float32), name='clst_time')

        # Training data - [index_for_vector, time]
        self.train_inputs = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='train_inputs')
        train_inputs = self.train_inputs
        # Labels for the training data
        self.train_labels = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='train_labels')
        train_labels = self.train_labels
        # Time for every pair (a, b)
        self.train_inputs_time = tf.placeholder(tf.float32, shape=[opts.batch_size, 1])
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
        rho_lup_cntx = tf.nn.embedding_lookup(self.rho, tf.reshape(train_inputs, [-1]))

        # Cluster time repeated batch_size times, for every sample in batch
        #we have all the cluster times
        t_ctime = tf.tile(self.clst_time, [opts.batch_size, 1])
        # Time for every word repated for number of clusters and flatened 
        t_wtime = tf.reshape(tf.tile(train_inputs_time, [1, opts.nclst]), [-1 ,1])
        # Time constraint repated batch_size times
        t_tau = tf.tile(self.tau, [opts.batch_size, 1])

        # Time difference between word and cluster time
        time_diff = -1 * tf.reshape(tf.abs(t_ctime - t_wtime) * tf.abs(t_tau), [-1, opts.nclst]) 
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
        selected_clst_cntx = tf.nn.embedding_lookup(self.cntx_clst, tf.reshape(selected_time_diff_ind, [-1])) 
        t_a_cntx = tf.reshape(tf.tile(train_inputs, [1, opts.clst_window]), [-1])
        selected_emb_cntx = tf.nn.embedding_lookup(cent_cntx, t_a_cntx)

        fc = tf.exp(tf.reshape(selected_time_diff, [-1, 1])) 
        gc_cntx = selected_clst_cntx - selected_emb_cntx

        dynamic_repr_cntx = tf.reduce_sum( tf.reshape(gc_cntx * tf.sigmoid(selected_rho_cntx) * fc, 
            [opts.batch_size, opts.emb_dim, opts.clst_window]), 2) 

        # The final representation (static + dynamic) of all contexts in the input dataset
        all_repr_cntx = cntx_emb + dynamic_repr_cntx

        ### Calculation of the word (label) representation
        rho_lup_word =  tf.nn.embedding_lookup(self.rho, tf.reshape(train_labels, [-1]))
        # We have to flatten the rho_lup to match the flat_selected_time_diff_ind
        flat_rho_lup_word = tf.reshape(rho_lup_word, [-1, 1])

        # Rhos for selected clusters based on time difference between word and clst_time
        selected_rho_word = tf.gather(flat_rho_lup_cntx, flat_selected_time_diff_ind)

        # Make t vars for calculating the transform function g_c(x_a)
        #selected_clst are cluster embeddings for every context in the input
        #t_a is all words repated M times, where M is the number of clusters used
        selected_clst_word = tf.nn.embedding_lookup(self.word_clst, tf.reshape(selected_time_diff_ind, [-1]))
        t_a_word = tf.reshape(tf.tile(train_labels, [1, opts.clst_window]), [-1])
        selected_emb_word = tf.nn.embedding_lookup(cent_word, t_a_word)

        gc_word = selected_clst_word - selected_emb_word

        dynamic_repr_word = tf.reduce_sum( tf.reshape(gc_word * tf.sigmoid(selected_rho_word) * fc,
            [opts.batch_size, opts.emb_dim, opts.clst_window]), 2)

        # The final representation (static + dynamic) of all contexts in the input dataset
        all_repr_word = word_emb + dynamic_repr_word


        ### Negative sampling
        labels_matrix = tf.reshape(tf.cast(train_labels, dtype=tf.int64), [opts.batch_size, 1])
        
        all_repr_neg = []
        for i in range(opts.nneg):
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
            rho_lup_neg = tf.nn.embedding_lookup(self.rho, tf.reshape(negative_ids, [-1]))

            # We have to flatten the rho_lup to match the flat_selected_time_diff_ind
            flat_rho_lup_neg = tf.reshape(rho_lup_neg, [-1, 1])

            # Rhos for selected clusters based on time difference between word and clst_time
            selected_rho_neg = tf.gather(flat_rho_lup_neg, flat_selected_time_diff_ind)

            # Make t vars for calculating the transform function g_c(x_a)
            #selected_clst are cluster embeddings for every context in the input
            #t_a is all words repated M times, where M is the number of clusters used
            selected_clst_neg = tf.nn.embedding_lookup(self.cntx_clst, tf.reshape(selected_time_diff_ind, [-1])) 
            t_a_neg = tf.reshape(tf.tile(negative_ids, [1, opts.clst_window]), [-1])
            selected_emb_neg = tf.nn.embedding_lookup(cent_cntx, t_a_neg)

            gc_neg = selected_clst_neg - selected_emb_neg

            dynamic_repr_neg = tf.reduce_sum( tf.reshape(gc_neg * tf.sigmoid(selected_rho_neg) * fc, 
                [opts.batch_size, opts.emb_dim, opts.clst_window]), 2) 

            # The final representation (static + dynamic) of all contexts in the input dataset
            neg_emb = tf.nn.embedding_lookup(cent_cntx, tf.reshape(negative_ids, [-1]))
            all_repr_neg.append(neg_emb + dynamic_repr_neg)

        # A hack allowing us to have more than one negative word per example.
        #t_all_repr_neg = tf.tile(all_repr_neg, [opts.nneg, 1])
        # THE REMOVE USED
        t_all_repr_neg = tf.concat(0, all_repr_neg)
        t_all_repr_word = tf.reshape(tf.tile(all_repr_word, [1, opts.nneg]), [opts.nneg * opts.batch_size, -1])

        t_negative_logits = tf.reduce_sum(tf.mul(t_all_repr_word, t_all_repr_neg), 1) #+ true_b
        negative_logits = tf.reshape(t_negative_logits, [-1, opts.nneg])

        true_logits = tf.reduce_sum(tf.mul(all_repr_word, all_repr_cntx), 1) #+ true_b

        return true_logits, negative_logits

    def generate_fullbatch(self, data, dataset_name='test'):
        opts = self._options
        data = DocumentTimePair(data, opts.time_transform, opts._start_time)
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

        # Generate a batch using all samples and all words from a sample
        for example in data:
            t = example[0]
            tmp = example[1]
            # Filter words in doc by probability 
            #TODO: None or nothing
            doc = [w if w in self._word2id and (data_params['probs'][self._word2id[w]] >= 1 or 
                data_params['probs'][self._word2id[w]] > np.random.rand()) else None for w in tmp]

            if len(doc) > 0:
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
        batch = np.reshape(np.array(batch), [-1, 1])
        labels = np.reshape(np.array(labels), [-1, 1])
        time = np.reshape(np.array(time), [-1, 1])

        return batch, labels, time 

    def generate_eval_batch(self, data_path):
        opts = self._options
        data = DocumentTimePair(data_path, opts.time_transform, opts._start_time)
        batch = []
        time = []

        for example in data:
            tmp = [self._word2id[x] for x in example[1] if x in self._word2id]
            if len(tmp) == 0:
                continue
            time.append(example[0])
            batch.append(tmp)

        batch = np.array(batch)
        time = np.array(time)
        return batch, time

    def generate_batch(self, data_path, dataset_name='train'):
        opts = self._options
        data_params = {}
        # Crate a data iterator over the whole dataset, no skips
        data = DocumentTimePair(data_path, opts.time_transform, opts._start_time)

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

        # Create new data iterator with select_probability
        data = DocumentTimePair(data_path, opts.time_transform, opts._start_time, select_prob)
        while True:
            for example in data:
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

                            if added_pairs >= opts.max_pairs_from_sample or n_added >= opts.epoch_size:
                                break
                        if added_pairs >= opts.max_pairs_from_sample or n_added >= opts.epoch_size:
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

    
    def _thread_target(self, queue):
        while True:
            data = queue.get()
            feed_dict={self.train_inputs: data[0], 
                self.train_labels: data[1],
                self.train_inputs_time: data[2],
                self.cent_word: self._cent_word,
                self.cent_cntx: self._cent_cntx,
                self.train_loss: self._train_loss,
                self.test_loss: self._test_loss}

            _, _loss = self._session.run([self._train_a2, self._loss], feed_dict)
            queue.task_done()
 

    def train2(self):
        print("Train started")
        opts = self._options

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(opts.save_path, self._session.graph)

        batch, labels, time = self.generate_batch(opts.train_data)
        loss = 0
        #Make Queue
        queue = Queue(maxsize=0)
        for i in range(opts.epoch_size // opts.batch_size):
            queue.put([batch[i*opts.batch_size:(i+1)*opts.batch_size],
                labels[i*opts.batch_size:(i+1)*opts.batch_size],
                time[i*opts.batch_size:(i+1)*opts.batch_size]])


        for j in range(opts.concurrent_steps):
            worker = Thread(target=self._thread_target, args=(queue,))
            worker.setDaemon(True)
            worker.start()

        queue.join()

        """
        self._train_loss = loss
        if epoch > 0:
            summary_str = self._session.run(summary_op, feed_dict)
            summary_writer.add_summary(summary_str, epoch)
        """


    def train(self):
        print("Train started")
        opts = self._options

        #eval_docs, eval_docs_time = self.generate_eval_batch(opts.train_data)
        batch, labels, time = self.generate_batch(opts.train_data)
        loss = 0
        for i in range(opts.epoch_size // opts.batch_size):
            feed_dict={self.train_inputs: batch[i*opts.batch_size:(i+1)*opts.batch_size], 
                self.train_labels: labels[i*opts.batch_size:(i+1)*opts.batch_size],
                self.train_inputs_time: time[i*opts.batch_size:(i+1)*opts.batch_size],
                self.cent_word: self._cent_word,
                self.cent_cntx: self._cent_cntx,
                self.epoch: self._epoch}
            if self._epoch % 2 == 0:
                _, _loss = self._session.run([self._train_a1, self._loss], feed_dict)
            else:
                _, _loss = self._session.run([self._train_a2, self._loss], feed_dict)
            
            loss += _loss
        
        loss = loss / float(opts.epoch_size // opts.batch_size)
        self._train_loss = loss
        """
        pred_time_error = 0
        i = 0
        for j in range(len(eval_docs)):
            feed_dict={self.eval_doc: eval_docs[j],
                 self.eval_doc_time: eval_docs_time[j]}
            pred_time_error += self._session.run(self.pred_time_error, feed_dict)
        pred_time_error = pred_time_error / len(eval_docs)
        print(pred_time_error)
        """

    def test(self):
        print("Test started")
        opts = self._options

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(opts.save_path, self._session.graph)

        
        if self._testset is None:
            self._testset = self.generate_batch(opts.test_data)
        batch, labels, time = self._testset
        #eval_docs, eval_docs_time = self.generate_eval_batch(opts.test_data)
        
        loss = 0
        for i in range(len(batch) // opts.batch_size):
            feed_dict={self.train_inputs: batch[i*opts.batch_size:(i+1)*opts.batch_size], 
                self.train_labels: labels[i*opts.batch_size:(i+1)*opts.batch_size],
                self.train_inputs_time: time[i*opts.batch_size:(i+1)*opts.batch_size],
                self.cent_word: self._cent_word,
                self.cent_cntx: self._cent_cntx,
                self.epoch: self._epoch}
 
            loss += self._session.run(self._loss, feed_dict) 
        loss = loss / float(len(batch) // opts.batch_size)
        self._test_loss = loss
        i = 0
        feed_dict={self.train_loss: self._train_loss,
             self.test_loss: self._test_loss,
             self.epoch: self._epoch}
        summary_str = self._session.run(summary_op, feed_dict)
        summary_writer.add_summary(summary_str, self._epoch)



    def optimize(self, loss):
        """Build the graph to optimize the loss function."""
        opts = self._options
        
        optimizer = tf.train.AdamOptimizer(opts.learning_rate, epsilon=1e-4)
        grads_and_vars = optimizer.compute_gradients(loss, gate_gradients=optimizer.GATE_NONE)

        self.gov = grads_and_vars
        
        train = optimizer.apply_gradients(grads_and_vars)
        
        self._train = train
        
        """
        optimizer = tf.train.AdamOptimizer(opts.learning_rate, epsilon=1e-4)
        train = optimizer.minimize(loss, gate_gradients=optimizer.GATE_NONE)
        self._train = train
        """

    def optimize_a1(self, loss):
        opts = self._options
        
        optimizer = tf.train.AdamOptimizer(opts.learning_rate, epsilon=1e-4)
        #lr = tf.maximum(0.0001, opts.learning_rate / tf.cast(tf.square(self.epoch + 1), tf.float32))
        #optimizer = tf.train.GradientDescentOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(loss, gate_gradients=optimizer.GATE_NONE)
        
        gav = []
        for ind in range(len(grads_and_vars)):
            pair = grads_and_vars[ind]
            if "rho" in pair[1].name:
                print(str(pair[1].name))
                gav.append(pair)

        train = optimizer.apply_gradients(gav)
        self._train_a1 = train


    def optimize_a2(self, loss):
        opts = self._options
        
        optimizer = tf.train.AdamOptimizer(opts.learning_rate, epsilon=1e-4)
        #lr = tf.maximum(0.0001, opts.learning_rate / tf.cast(tf.square(self.epoch + 1), tf.float32))
        #optimizer = tf.train.GradientDescentOptimizer(lr)
 
        grads_and_vars = optimizer.compute_gradients(loss, gate_gradients=optimizer.GATE_NONE)
        self.gav = grads_and_vars
        
        gav = []
        for ind in range(len(grads_and_vars)):
            pair = grads_and_vars[ind]
            if "rho" not in pair[1].name:
                print(str(pair[1].name))
                gav.append(pair)

        train = optimizer.apply_gradients(gav)
        self._train_a2 = train



    def nce_loss(self, true_logits, negative_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        #true_logits_clipped = tf.clip_by_value(true_logits, 2, -100, 100)
        #negative_logits_clipped = tf.clip_by_value(negative_logits, 2, -100, 100)
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


def mm_prob(prob, lmbd=0.9999):
    return lmbd * prob + (1 - lmbd) * (1 - prob)

def cluster_docs(rho, data, word2id, clst_time, tau):
    clsts = []

    for sample in data:
        time = sample[0]
        doc = sample[1]
        
        time_diff = np.reshape(np.exp( -np.abs((clst_time - time) * tau) ), -1)
        prob = None
        for word in doc:
            if word not in word2id:
                continue
            if prob is None:
                tmp = (1 / 1 + np.exp(-rho[word2id[word], :])) * time_diff
                tmp[tmp == 0] = 0.000000001
                prob = np.log(tmp)
            else:
                tmp = (1 / 1 + np.exp(-rho[word2id[word], :])) * time_diff
                tmp[tmp == 0] = 0.000000001
                prob = prob + np.log(tmp)


        if prob is None:
            clsts.append(0)
        else:
            prob = np.array(prob)
            clsts.append(np.argmax(prob))

    return clsts

def test_clustering(rho, data, word2id, clst_time, tau, tags_clst_files, nclst, beta, iter, statistics_file=None):
    twe_clst = cluster_docs(rho, data, word2id, clst_time, tau)

    for clst_file in tags_clst_files:
         matrix = vmeasure.tags_twc(clst_file, twe_clst, nclst)

         (h, c, vms) = vmeasure.calc_vmeasure(matrix, beta)
         print("H C VMS: {} {} {}".format(h, c, vms))
         if statistics_file is not None:
             f = open(statistics_file, 'a')
             f.write("{},{},{},{}\n".format(iter, h, c, vms))
             f.close()

def test_time_pred(rho, data, word2id, clst_time, tau):
    error = 0
    cnt = 0
    for sample in data:
        time = sample[0]
        doc = sample[1]
        prob = None
        for word in doc:
            if word not in word2id:
                continue

            if prob is None:
                prob = np.log(1 / (1 + np.exp(-rho[word2id[word], :])))
            else:
                prob = prob + np.log(1 / (1 + np.exp(-rho[word2id[word], :])))
        if prob is not None:
            prob[np.isnan(prob)] = -10e9
            z = np.max(prob)
            if np.sum(prob) != 0:
                pred = np.sum(np.reshape(clst_time, -1) * np.exp(prob - z)) / np.sum(np.exp(prob - z))

                error += np.abs(pred - time)
                cnt += 1
    
    print("TIME: {} - {}".format(error / cnt, cnt))

def words_in_cluster(rho, id2word, topw=20):
    rho = 1 / (1 + np.exp(-rho))
    
    top_ind = []
    for cind in range(rho.shape[1]):
        probs = rho[:, cind] / np.sum(rho[:, [x for x in range(rho.shape[1]) if x != cind]], 1)
        top_ind.append(np.argsort(-probs)[0:topw])

    return top_ind



def main(opts=None):
    if opts is None:
        opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            twe = TempWordEmb(opts, session)
            for i in range(opts.nepochs):
                twe._epoch = i
                print("Started epoch: {}".format(i))
                if i % 30 == 0:
                    #Save the tf model
                    twe.saver.save(twe._session, opts.save_path + "model", global_step=i)

                    #Test clustering
                    rho = twe.rho.eval()
                    clst_time = twe.clst_time.eval()
                    tau = twe.tau.eval()
                    tags_clst_files = opts.tags_clst_files
                    nclst = opts.nclst
                    beta = opts.vm_beta
                    iter = i
                    statistics_file = opts.save_path + "statistics.txt"
                    data = DocumentTimePair(opts.train_data, opts.time_transform, opts._start_time)

                    if tags_clst_files is not None:
                        test_clustering(rho, data, twe._word2id, clst_time, tau, tags_clst_files, nclst, beta,
                                iter, statistics_file)

                    test_time_pred(rho, data, twe._word2id, clst_time, tau)

                    #Print top words for clusters
                    rho_exp = 1 / (1 + np.exp(-rho))
                    """
                    for irow, row in enumerate(np.transpose(rho_exp)):
                        row = np.argsort(-row)
                        print("-------------- CLST: {} ---------------".format(irow))

                        for ind in row[0:40]:
                            print(twe._id2word[ind])
                    """
                twe.train()
                twe.test()
            print("--------DONE-------")


