import tensorflow as tf
import argparse, json, sys, os
from util.util import pgbar

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	parser.add_argument('--use_word2vec', type=str, default='true', choices=['true', 'false'])
	args = parser.parse_args()
	word2vec_dim = 5 # fix dimension to 100 for reduce calculation speed
	vocab_dim = 000
	latent_dim = 6 # must be even
	max_sent_len = 4

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.float32, [None, max_sent_len, word2vec_dim])
	X_len = tf.placeholder(tf.int32, [None])
	Y = tf.placeholder(tf.float32, [None, max_sent_len, word2vec_dim])

	# cell_enc_fw = tf.nn.rnn_cell.LSTMCell(num_units=latent_dim//2)
	# cell_enc_bw = tf.nn.rnn_cell.LSTMCell(num_units=latent_dim//2)
	# outputs_enc, state_enc = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_enc_fw, cell_bw=cell_enc_bw, dtype=tf.float32, sequence_length=X_len, inputs=X, scope='g1')
	# outputs_fw = tf.transpose(outputs_enc[0], [1, 0, 2])
	# outputs_bw = tf.transpose(outputs_enc[1], [1, 0, 2])
	# latent_z = tf.concat([outputs_fw[-1], outputs_bw[-1]], axis=1)
	cell_enc = tf.nn.rnn_cell.LSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, dtype=tf.float32, sequence_length=X_len, inputs=X, scope='g1')
	latent_z = state_enc.h
	# latent_z = tf.reshape(tf.slice(outputs_enc, [0, max_sent_len-1, 0], [-1, 1, -1]), [-1, latent_dim])
	latent_pad = tf.reshape(tf.tile(latent_z * 0.0, [max_sent_len-1, 1]), [-1, max_sent_len-1, latent_dim])
	latent_reshape = tf.reshape(latent_z, [-1, 1, latent_dim])
	x_inner = tf.concat([latent_reshape, latent_pad], axis=1)
	cell_dec = tf.nn.rnn_cell.LSTMCell(num_units=word2vec_dim)
	outputs_dec, state_dec = tf.nn.dynamic_rnn(cell=cell_dec, dtype=tf.float32, sequence_length=X_len, inputs=x_inner, scope='g2')
	logits = tf.nn.softmax(outputs_dec)
	# weights = tf.ones([1, max_sent_len])
	# loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=logits, targets=Y, weights=weights))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	train = tf.train.AdamOptimizer(0.001).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(1, 3001):
			feed_dict={X:[[[.1, .2, .3, .3, .1], [.2, .1, .2, .1, .4], [.3, .4, .1, .1, .1], [.0, .0, .0, .0, .0]]], X_len:[3], Y:[[[.1, .2, .3, .3, .1], [.2, .1, .2, .1, .4], [.3, .4, .1, .1, .1], [.0, .0, .0, .0, .0]]]}
			_, cost = sess.run([train, loss], feed_dict=feed_dict)
			if epoch % 100 == 0:
				print('===== %d =====' % epoch)
				print('loss : %f' % cost)
				print(sess.run(outputs_dec, feed_dict=feed_dict))


	#################### main logic ####################

	if args.mode == 'train':
		print('train? comming soon...')
	elif args.mode == 'test':
		print('test? comming soon...')
