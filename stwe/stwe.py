import tensorflow as tf
import numpy as np
import gensim


class STWE(object):
    def __init__(self, options, start_time=0, end_time=0):
        self.opts = options

    def add_placeholders(self):
        self.inputs = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[opts.batch_size, 1], name='inputs')
        self.time = tf.placeholder(tf.float32, shape=[opts.batch_size, 1], name='time')

        self.cent_word_emb = tf.placeholder(tf.float32,
                shape=[opts.vocab_size, opts.emb_dim], name='cent_word_emb')


    def create_feed_dict(self, inputs_batch, labels_batch, time_batch, cent_word_emb):
        feed_dict = {self.inputs: inputs_batch,
                     self.labels: labels_batch,
                     self.time: time_batch,
                     self.cent_word_emb = cent_word_emb}


    def add_prediction_op(self):
        opts = self.opts

        rho = tf.get_variable(name='rho',
                shape=(opts.vocab_size, opts.nclst),
                dtype=tf.float32,
                initializer=tf.ones_initializer()) * -5
        tau = tf.get_variable(name='tau',
                shape=(opts.nclst, 1),
                dtype=tf.float32,
                initializer=tf.ones_initializer())
        clusters_emb = tf.get_vairable(name='clusters_emb',
                shape=(opts.nclst, opts.emb_dim),
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.5, 0.5))
        cluster_time = tf.get_vairable(name='clusters_time',
                shape=(opts.nclst, 1),
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(opts.start_time, opts.end_time))


        # Get the embedding loopk-up for inputs and labels
        labels_emb = tf.nn.embedding_lookup(self.cent_word_emb, self.labels)
        inputs_emb = tf.nn.embedding_lookup(self.cent_word_emb, self.inputs)

        # Get the loopk up for rho, as rho is of shpape [vocab x nclst] this 
        #is pretty much same as embedding
        inputs_rho_lup = tf.nn.embedding_lookup(self,rho, self.inputs)

        # Get the time difference between each input sample and every cluster 
        #out shape is: batch_size x  #clusters
        time_diff = -1 * tf.abs(self.time - tf.transpose(cluster_time))
        # For each row find the K closest clusters, or the ones with the 
        #smallest time difference
        #out shape is: batch_size x opts.clst_window
        selected_time_diff, selected_time_diff_ind = tf.nn.top_k(time_diff, k=opts.clst_window)

        # For each selected cluster get rho
        #out shape is: batch_size x opts.clst_window
        selected_rho = tf.squeeze(tf.gather(rho, selected_time_diff_ind, axis=0), axis=[2])

        # Selected clusters embeddings
        #out shape is: batch_size x opts.clst_window x opts.emb_dim
        selected_clusters_emb = tf.nn.embedding_lookup(clusters_emb, selected_time_diff_ind)
        # Calculate g_c(x_a), the vector transformation function. From selected clusters
        #for one word subtract that word
        g_c = selected_clusters_emb - inputs_emb
