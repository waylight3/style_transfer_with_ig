import tensorflow as tf
import argparse, random, random, json, sys, os
from util.util import pgbar, prints, print_table
from nltk.tokenize import sent_tokenize, word_tokenize

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	parser.add_argument('--latent_dim', type=int, default=100, help='must be even')
	parser.add_argument('--max_sent_len', type=int, default=20, help='maximum number of words in each sentence')
	parser.add_argument('--batch_size', type=int, default=20, help='size of each batch. prefer to be a multiple of data size')
	parser.add_argument('--learning_rate', type=float, default=0.0003)
	args = parser.parse_args()
	
	mode = args.mode

	UNKOWN = 0
	GO = 1
	END = 2

	word2vec = {}
	word2idx = {}
	idx2word = {UNKOWN:'<UNK>', GO:'<GO>', END:'<END>'}
	word2vec_dim = 0
	vocab_dim = 3 # zero(UNKOWN), one(GO), two(END) is used for symbol
	latent_dim = args.latent_dim
	max_sent_len = args.max_sent_len
	batch_size = args.batch_size
	learning_rate = args.learning_rate
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
	embedding_dim = 100
	
	data_x = []
	data_x_len = []
	data_y = []
	data_size = 0

	with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
		lines = fp.read().strip().split('\n')
		for line in pgbar(lines, pre='[train.data]'):
			sent_idx = []
			temp = json.loads(line)
			words = word_tokenize(temp['text'])
			for word in words:
				if word in word2idx:
					sent_idx.append(word2idx[word])
				else:
					sent_idx.append(UNKOWN)
				if len(sent_idx) >= max_sent_len:
					break
			while len(sent_idx) < max_sent_len:
				sent_idx.append(END)
			data_x.append([GO] + sent_idx)
			data_y.append(sent_idx + [END])
			data_x_len.append(len(sent_idx) + 1)
			data_size += 1
	max_sent_len += 1

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.int32, [None, max_sent_len]) # +1 for <GO>
	X_len = tf.placeholder(tf.int32, [None])
	Y = tf.placeholder(tf.int32, [None, max_sent_len]) # +1 for <END>

	# X = tf.placeholder(tf.float64, [None, max_sent_len, word2vec_dim])
	# X_len = tf.placeholder(tf.int64, [None])
	# Y = tf.placeholder(tf.int64, [None, max_sent_len])

	init = tf.contrib.layers.xavier_initializer()
	embedding = tf.get_variable('embedding', shape=[vocab_dim, embedding_dim], initializer=init, dtype=tf.float32)
	inputs_enc = tf.nn.embedding_lookup(embedding, X)
	cell_enc = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, inputs=inputs_enc, sequence_length=X_len, dtype=tf.float32, scope='g1')
	cell_dec = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim, state_is_tuple=False)
	helper = tf.contrib.seq2seq.TrainingHelper(tf.zeros_like(inputs_enc), X_len)
	init = tf.concat([state_enc.h, state_enc.c], axis=-1)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_dec, helper=helper, initial_state=init)
	outputs_dec, last_state, last_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, impute_finished=True, maximum_iterations=max_sent_len)
	weights = tf.ones(shape=[batch_size, max_sent_len])
	loss = tf.contrib.seq2seq.sequence_loss(logits=outputs_dec.rnn_output, targets=Y, weights=weights)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)


	# cell_enc = tf.nn.rnn_cell.LSTMCell(num_units=latent_dim)
	# outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, dtype=tf.float64, sequence_length=X_len, inputs=X, scope='g1')
	# latent_z = state_enc.h
	# latent_pad = tf.reshape(tf.tile(latent_z * 0.0, [max_sent_len-1, 1]), [-1, max_sent_len-1, latent_dim])
	# latent_reshape = tf.reshape(latent_z, [-1, 1, latent_dim])
	# x_inner = tf.concat([latent_reshape, latent_pad], axis=1)
	# cell_dec = tf.nn.rnn_cell.LSTMCell(num_units=vocab_dim)
	# outputs_dec, state_dec = tf.nn.dynamic_rnn(cell=cell_dec, dtype=tf.float64, sequence_length=X_len, inputs=x_inner, scope='g2')
	# logits = outputs_dec
	# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
	# train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	# predict = tf.argmax(logits, axis=2)
	# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), tf.float64))


	#################### main logic ####################
	if mode == 'train':
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(1, 11):
				for batch in range(data_size // batch_size):
					feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size]}
					_, now_loss, test = sess.run([train, loss, inputs_enc], feed_dict=feed_dict)
					print(epoch, batch, now_loss)
					print(test)
					# if batch % 1 == 0:
					# 	print_data = []
					# 	print_data.append(['loss', now_loss])
					# 	print_data.append([])
					# 	for i in range(5):
					# 		test_idx = random.randrange(data_size)
					# 		ret = sess.run(outputs_dec, feed_dict={X:data_x[test_idx:test_idx+1], X_len:data_x_len[test_idx:test_idx+1], Y:data_y[test_idx:test_idx+1]})
					# 		print_data.append(['original', ' '.join([idx2word[idx] for idx in data_y[test_idx]])])
					# 		print_data.append(['predict', ' '.join([idx2word[idx] for idx in ret.sample_id[0]])])
					# 		if i != 4: print_data.append([])
					# 	print('\n')
					# 	print_table(print_data, title='%d epoch / %d batch' % (epoch, batch + 1), min_width=os.get_terminal_size()[0] - 1)
					# 	print()

		# with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
		# 	lines = fp.read().strip().split('\n')
		# 	for line in pgbar(lines, pre='[train.data]'):
		# 		sent_vec = []
		# 		sent_idx = []
		# 		temp = json.loads(line)
		# 		words = word_tokenize(temp['text'])
		# 		for word in words:
		# 			if word in word2vec:
		# 				sent_vec.append(word2vec[word])
		# 				sent_idx.append(word2idx[word])
		# 			else:
		# 				sent_vec.append([0.0 for i in range(word2vec_dim)])
		# 				# sent_idx.append(UNKOWN)
		# 				sent_idx.append(random.randrange(4000))
		# 			if len(sent_vec) >= max_sent_len:
		# 				break
		# 		data_x_len.append(len(sent_vec))
		# 		while len(sent_vec) < max_sent_len:
		# 			sent_vec.append([0.0 for i in range(word2vec_dim)])
		# 			sent_idx.append(END)
		# 		data_x.append(sent_vec)
		# 		data_y.append(sent_idx)
		# 		data_size += 1

		# data_idx = [i for i in range(data_size)]

		# with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# 	for epoch in range(1, 11):
		# 		# shuffle data order at each epoch for remove order-bias
		# 		# random.shuffle(data_idx)
		# 		for batch in range(data_size // batch_size):
		# 			feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size]}
		# 			_, now_loss, now_acc = sess.run([train, loss, accuracy], feed_dict=feed_dict)
		# 			if batch % 1 == 0:
		# 				print_data = []
		# 				print_data.append(['loss', now_loss])
		# 				print_data.append(['acc', now_acc])
		# 				print_data.append([])
		# 				for i in range(5):
		# 					test_idx = random.randrange(data_size)
		# 					ret = sess.run(predict, feed_dict={X:data_x[test_idx:test_idx+1], X_len:data_x_len[test_idx:test_idx+1], Y:data_y[test_idx:test_idx+1]})
		# 					print_data.append(['original', ' '.join([idx2word[idx] for idx in data_y[test_idx]])])
		# 					print_data.append(['predict', ' '.join([idx2word[idx] for idx in ret[0]])])
		# 					if i != 4: print_data.append([])
		# 				print('\n')
		# 				print_table(print_data, title='%d epoch / %d batch' % (epoch, batch + 1), min_width=os.get_terminal_size()[0] - 1)
		# 				print()


	elif mode == 'test':
		print('test? comming soon...')
