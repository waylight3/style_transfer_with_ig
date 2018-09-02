import tensorflow as tf
import argparse, random, json, sys, os
from util.util import pgbar
from nltk.tokenize import sent_tokenize, word_tokenize

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	args = parser.parse_args()
	
	mode = args.mode

	UNKOWN = 0

	word2vec = {}
	word2idx = {}
	idx2word = {UNKOWN:'?'}
	word2vec_dim = 0 # fix dimension to 100 for reduce calculation speed
	vocab_dim = 1 # zero(0) is used for UNKOWN symbol
	latent_dim = 10 # must be even
	max_sent_len = 20
	batch_size = 100 # prefer to be a multiple of data size
	with open('seq2seq2/data/word2vec.data', encoding='utf-8') as fp:
		lines = fp.read().strip().split('\n')
		for line in pgbar(lines, pre='[word2vec.data]'):
			temp = line.split()
			word = temp[0]
			vec = list(map(float, temp[1:]))
			word2vec[word] = vec
			word2vec_dim = len(vec)
			word2idx[word] = vocab_dim
			idx2word[vocab_dim] = word
			vocab_dim += 1

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.float64, [None, max_sent_len, word2vec_dim])
	X_len = tf.placeholder(tf.int64, [None])
	Y = tf.placeholder(tf.int64, [None, max_sent_len])

	cell_enc = tf.nn.rnn_cell.LSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, dtype=tf.float64, sequence_length=X_len, inputs=X, scope='g1')
	latent_z = state_enc.h
	latent_pad = tf.reshape(tf.tile(latent_z * 0.0, [max_sent_len-1, 1]), [-1, max_sent_len-1, latent_dim])
	latent_reshape = tf.reshape(latent_z, [-1, 1, latent_dim])
	x_inner = tf.concat([latent_reshape, latent_pad], axis=1)
	cell_dec = tf.nn.rnn_cell.LSTMCell(num_units=vocab_dim)
	outputs_dec, state_dec = tf.nn.dynamic_rnn(cell=cell_dec, dtype=tf.float64, sequence_length=X_len, inputs=x_inner, scope='g2')
	logits = outputs_dec
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	train = tf.train.AdamOptimizer(0.001).minimize(loss)
	predict = tf.argmax(logits, axis=2)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float64))


	#################### main logic ####################
	if mode == 'train':
		data_x = []
		data_x_len = []
		data_y = []
		data_size = 0

		with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[train.data]'):
				sent_vec = []
				sent_idx = []
				temp = json.loads(line)
				words = word_tokenize(temp['text'])
				for word in words:
					if word in word2vec:
						sent_vec.append(word2vec[word])
						sent_idx.append(word2idx[word])
					else:
						sent_vec.append([0.0 for i in range(word2vec_dim)])
						sent_idx.append(UNKOWN)
				data_x_len.append(len(sent_vec))
				while len(sent_vec) < max_sent_len:
					sent_vec.append([0.0 for i in range(word2vec_dim)])
					sent_idx.append(UNKOWN)
				data_x.append(sent_vec)
				data_y.append(sent_idx)
				data_size += 1

		data_idx = [i for i in range(data_size)]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(1, 11):
				# shuffle data order at each epoch for remove order-bias
				# random.shuffle(data_idx)
				for batch in range(data_size // batch_size):	
					feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size]}
					_, now_loss, now_acc = sess.run([train, loss, accuracy], feed_dict=feed_dict)
					if batch % 1 == 0:
						print('===== %d / %d =====' % (epoch, batch))
						print('loss : %f' % now_loss)
						print('acc  : %f' % now_acc)
						test_idx = random.randrange(data_size)
						ret = sess.run(predict, feed_dict={X:data_x[test_idx:test_idx+1], X_len:data_x_len[test_idx:test_idx+1], Y:data_y[test_idx:test_idx+1]})
						print('original : ' + ' '.join([idx2word[idx] for idx in data_y[test_idx]]))
						print('predict  : ' + ' '.join([idx2word[idx] for idx in ret[0]]))


	elif mode == 'test':
		print('test? comming soon...')
