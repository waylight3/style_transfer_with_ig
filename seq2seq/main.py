import tensorflow as tf
import numpy as np


######### parameters ###########

word2vec_dim = 100
rnn_step_size = 100
total_epoch = 10
batch_size = 64
learning_rate = 0.0001
n_hidden = 128
n_latent = 10


######### model start ##########

tf.reset_default_graph()


X_in = tf.placeholder(dtype = tf.float32, shape = [None, word2vec_dim, rnn_step_size], name = 'X')
W_mn = tf.Variable(tf.random_normal([n_hidden, n_latent]))
b_mn = tf.Variable(tf.random_normal([n_latent]))
W_sd = tf.Variable(tf.random_normal([n_hidden, n_latent]))
b_sd = tf.Variable(tf.random_normal([n_latent]))
keep_prob = tf.placeholder(dtype = tf.float32, shape = (), name = 'keep_prob')

def encoder(X_in, keep_prob) :
	with tf.variable_scope("encoder", reuse = None) :
		cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
		outputs, state = tf.nn.dynamic_rnn(cell = cell, dtype = float64, inputs = X_in)
		outputs = tf.transpose(outputs, [1, 0, 2])
		outputs = outputs[-1]

		mn = tf.matmul(outputs, W_mn) + b_mn
		sd = tf.matmul(outputs, W_sd) + b_sd
		ep = tf.random_normal(tf.stack([tf.shape(X_in)[0], n_latent]))
		z = mn + tf.multiply(ep, tf.exp(sd))

		return z, mn, sd, state

def decoder(sampled_z, enc_state, keep_prob) :
	with tf.variable_scope("decode", reuse = None) :
		cell = tf.nn.rnn_cell.LSTMCell(num_units = n_hidden)
		outputs, state = tf.nn.dynamic_rnn

