import tensorflow as tf
import numpy as np
import gensim


class STWE(object):
    def __init__(self, options, vocab):
        self.opts = options
        self.vocab = vocab

    def add_placeholders(self):
        opts = self.opts
        self.inputs = tf.placeholder(tf.int64, shape=[opts.batch_size, 1], name='inputs')
        self.labels = tf.placeholder(tf.int64, shape=[opts.batch_size, 1], name='inputs')
        self.time = tf.placeholder(tf.float32, shape=[opts.batch_size, 1], name='time')

        self.cent_word_emb = tf.placeholder(tf.float32,
                shape=[len(self.vocab), opts.emb_dim], name='cent_word_emb')


    def create_feed_dict(self, inputs_batch, labels_batch, time_batch, cent_word_emb):
        feed_dict = {self.inputs: inputs_batch,
                     self.labels: labels_batch,
                     self.time: time_batch,
                     self.cent_word_emb: cent_word_emb}


    def model(self):
        opts = self.opts

        rho = tf.get_variable(name='rho',
                shape=(len(self.vocab), opts.nclst),
                dtype=tf.float32,
                initializer=tf.ones_initializer()) * -5
        tau = tf.get_variable(name='tau',
                shape=(opts.nclst, 1),
                dtype=tf.float32,
                initializer=tf.ones_initializer())
        clusters_emb = tf.get_variable(name='clusters_emb',
                shape=(opts.nclst, opts.emb_dim),
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.5, 0.5))
        cluster_time = tf.get_variable(name='clusters_time',
                shape=(opts.nclst, 1),
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(opts.start_time, opts.end_time))

        # Negative sampling
        negative_ids, _, _ = (tf.nn.log_uniform_candidate_sampler(
            true_classes=self.labels,
            num_true=1,
            num_sampled=opts.batch_size * opts.nneg,
            unique=True,
            range_max=len(self.vocab)))

        # Reshape to a [x, 1] tensor
        negative_ids = tf.reshape(negative_ids, [-1, 1])

        # Get the embedding loopk-up for inputs and labels
        labels_emb_static = tf.nn.embedding_lookup(self.cent_word_emb, tf.reshape(self.labels, [-1]))
        inputs_emb_static = tf.nn.embedding_lookup(self.cent_word_emb, tf.reshape(self.inputs, [-1]))
        negative_emb_static = tf.nn.embedding_lookup(self.cent_word_emb, tf.reshape(negative_ids, [-1]))

        # Get the look up for rho, as rho is of shape [vocab x nclst] this 
        #is pretty much same as embedding
        #out shape is: batch x nclst
        inputs_rho_lup = tf.nn.embedding_lookup(rho, tf.reshape(self.inputs, [-1]))
        labels_rho_lup = tf.nn.embedding_lookup(rho, tf.reshape(self.labels, [-1]))
        negative_rho_lup = tf.nn.embedding_lookup(rho, tf.reshape(negative_ids, [-1]))

        # Get the time difference between each input sample and every cluster, multiply
        # the time difference by tau (don't allow tau to be negative - test this TODO).
        #out shape is: batch_size x  #clusters
        time_diff = -1 * tf.abs(self.time - tf.transpose(cluster_time)) * tf.abs(tf.transpose(tau))
        # For each row find the K closest clusters, or the ones with the 
        #smallest time difference
        #out shape is: batch_size x opts.clst_window
        selected_time_diff, selected_time_diff_ind = tf.nn.top_k(time_diff, k=opts.clst_window)

        # For negative we take the same time stamps as for the inputs, but have to tile them
        #in the amount of nneg
        selected_time_diff_neg = tf.tile(selected_time_diff, [opts.nneg, 1])
        selected_time_diff_ind_neg = tf.tile(selected_time_diff_ind, [opts.nneg, 1])


        # For each selected cluster get rho
        #out shape is: batch_size x opts.clst_window
        _ind_help = tf.expand_dims(
                tf.tile(
                    tf.reshape(tf.range(0, opts.batch_size), [-1, 1]), 
                    [1, opts.clst_window]),
                -1)
        #_ind_help_neg = tf.tile(_ind_help, [opts.nneg, 1])
        _rho_inp_indexer = tf.concat([_ind_help, tf.expand_dims(selected_time_diff_ind, -1)], 2)

        _rho_neg_indexer = tf.tile(_rho_inp_indexer, [opts.nneg, 1, 1])
        print("HERE")
        print(selected_time_diff_ind.shape)
        print(inputs_rho_lup.shape)
        selected_rho_inputs = tf.gather_nd(inputs_rho_lup, _rho_inp_indexer)
        print(selected_rho_inputs.shape)
        selected_rho_labels = tf.gather_nd(labels_rho_lup, _rho_inp_indexer)
        print(selected_rho_labels.shape)

        selected_rho_negative = tf.gather_nd(negative_rho_lup, _rho_neg_indexer)

        # Selected clusters embeddings
        #out shape is: batch_size x opts.clst_window x opts.emb_dim
        selected_clusters_emb = tf.nn.embedding_lookup(clusters_emb, selected_time_diff_ind)
        selected_clusters_emb_neg = tf.nn.embedding_lookup(clusters_emb, selected_time_diff_ind_neg)

        # Calculate g_c(x_a), the vector transformation function. From selected clusters
        #for one word subtract that word
        #out shape is: batch_size x opts.clst_window x opts.emb_dim
        g_c_inputs = selected_clusters_emb - tf.expand_dims(inputs_emb_static, 1)
        g_c_labels = selected_clusters_emb - tf.expand_dims(labels_emb_static, 1)
        g_c_negative = selected_clusters_emb_neg - tf.expand_dims(negative_emb_static, 1)
        print(g_c_inputs.shape)

        # Calculate the f_c(t) - time limiter function, it is just the exponent of the
        #selected_time_diff
        #out shape is: batch_size x opts.clst_window
        f_c = tf.exp(selected_time_diff)
        f_c_negative = tf.exp(selected_time_diff_neg)

        # Dynamic representation for words in self.inputs
        #out shape is: batch_size x opts.emb_dim
        inputs_emb_dynamic = g_c_inputs * tf.expand_dims(selected_rho_inputs, -1)
        print(inputs_emb_dynamic.shape)
        inputs_emb_dynamic = inputs_emb_dynamic * tf.expand_dims(f_c, -1)
        inputs_emb_dynamic = tf.reduce_sum(inputs_emb_dynamic, axis=1)
        # Full representation for inputs embedding
        inputs_emb = inputs_emb_static + inputs_emb_dynamic

        # Dynamic representation for words in self.labels
        #out shape is: batch_size x opts.emb_dim
        labels_emb_dynamic = g_c_labels * tf.expand_dims(selected_rho_labels, -1)
        labels_emb_dynamic = labels_emb_dynamic * tf.expand_dims(f_c, -1)
        labels_emb_dynamic = tf.reduce_sum(labels_emb_dynamic, axis=1)
        # Full representation for labels embedding
        labels_emb = labels_emb_static + labels_emb_dynamic


        # Dynamic representation for words in negative
        #out shape is: (batch_size * self.opts.nneg) x opts.emb_dim
        negative_emb_dynamic = g_c_negative * tf.expand_dims(selected_rho_negative, -1)
        negative_emb_dynamic = negative_emb_dynamic * tf.expand_dims(f_c_negative, -1)
        negative_emb_dynamic = tf.reduce_sum(negative_emb_dynamic, axis=1)
        # Full representation for inputs embedding
        negative_emb = negative_emb_static + negative_emb_dynamic

        # Tile the labels for nneg times to match the negative_emb
        _labels_emb_nneg = tf.tile(labels_emb, [opts.nneg, 1])

        true_logits = tf.reduce_sum(tf.matmul(labels_emb, inputs_emb, transpose_b=True), 1)
        negative_logits = tf.reduce_sum(tf.matmul(_labels_emb_nneg, negative_emb, transpose_b=True), 1)

        return true_logits, negative_logits
