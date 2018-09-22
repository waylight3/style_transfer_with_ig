import tensorflow as tf
import argparse, random, random, json, sys, os
from util.util import pgbar, prints, print_table
from nltk.tokenize import sent_tokenize, word_tokenize
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib import layers
import matplotlib.pyplot as plt
import win_unicode_console

win_unicode_console.enable()
tf.set_random_seed(2015147554)

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'])
	parser.add_argument('--latent_dim', type=int, default=100)
	parser.add_argument('--embedding_dim', type=int, default=100)
	parser.add_argument('--max_sent_len', type=int, default=20, help='maximum number of words in each sentence')
	parser.add_argument('--batch_size', type=int, default=50, help='size of each batch. prefer to be a multiple of data size')
	parser.add_argument('--learning_rate', type=float, default=0.01)
	parser.add_argument('--total_epoch', type=int, default=15)
	parser.add_argument('--use_word2vec', type=str, default='false', choices=['true', 'false'])
	args = parser.parse_args()
	
	mode = args.mode

	UNKOWN = 0
	GO = 1
	END = 2

	word2vec = {}
	word2idx = {}
	idx2word = {UNKOWN:'?', GO:'<GO>', END:''}
	word2vec_dim = 0
	word_list = []
	vocab_dim = 3 # zero(UNKOWN), one(GO), two(END) is used for symbol
	latent_dim = args.latent_dim
	max_sent_len = args.max_sent_len
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	total_epoch = args.total_epoch
	use_word2vec = True if args.use_word2vec == 'true' else False

	### for word2vec mode
	if use_word2vec:
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
	### for non-word2vec mode
	else:
		with open('seq2seq2/data/word_list.data', encoding='utf-8') as fp:
			words = fp.read().strip().split('\n')
			for word in pgbar(words, pre='[word_list.data]'):
				word_list.append(word)
				word2idx[word] = vocab_dim
				idx2word[vocab_dim] = word
				vocab_dim += 1
		embedding_dim = args.embedding_dim

	data_x = []
	data_x_len = []	
	data_y = []
	data_mask = []
	data_size = 0

	### for word2vec mode
	if use_word2vec:
		with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[train.data]'):
				sent_idx = []
				sent_vec = []
				sent_mask = []
				temp = json.loads(line)
				words = temp['text'].split()
				for word in words:
					if word in word2idx:
						sent_idx.append(word2idx[word])
						sent_vec.append(word2vec[word])
						sent_mask.append(1.0)
					else:
						sent_idx.append(UNKOWN)
						sent_vec.append([0.0 for _ in range(word2vec_dim)])
						sent_mask.append(0.0)
					if len(sent_idx) >= max_sent_len:
						break
				while len(sent_idx) < max_sent_len:
					sent_idx.append(END)
					sent_vec.append([0.0 for _ in range(word2vec_dim)])
					sent_mask.append(0.0)
				data_x.append([GO] + sent_idx)
				# data_x.append([0.0 for _ in range(word2vec_dim)] + sent_vec)
				data_y.append(sent_idx + [END])
				data_mask.append(sent_mask + [0.0])
				data_x_len.append(len(sent_idx) + 1)
				data_size += 1
	### for non-word2vec mode
	else:
		with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[train.data]'):
				sent_idx = []
				sent_mask = []
				temp = json.loads(line)
				words = temp['text'].split()
				for word in words:
					if word in word2idx:
						sent_idx.append(word2idx[word])
						sent_mask.append(1.0)
					else:
						sent_idx.append(UNKOWN)
						sent_mask.append(0.0)
					if len(sent_idx) >= max_sent_len:
						break
				while len(sent_idx) < max_sent_len:
					sent_idx.append(END)
					sent_mask.append(0.0)
				data_x.append([GO] + sent_idx)
				data_y.append(sent_idx + [END])
				data_mask.append(sent_mask + [0.0])
				data_x_len.append(len(sent_idx) + 1)
				data_size += 1

	max_sent_len += 1 # +1 for <GO> and <END>

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.int32, [None, max_sent_len])
	# X = tf.placeholder(tf.float32, [None, max_sent_len, word2vec_dim])
	X_len = tf.placeholder(tf.int32, [None])
	Y = tf.placeholder(tf.int32, [None, max_sent_len])
	Y_len = tf.placeholder(tf.int32, [None])
	Y_mask = tf.placeholder(tf.float32, [None, max_sent_len])

	# init = tf.contrib.layers.xavier_initializer()
	# embedding = tf.get_variable('embedding', shape=[vocab_dim, embedding_dim], initializer=init, dtype=tf.float32)
	# inputs_enc = tf.nn.embedding_lookup(embedding, X)
	inputs_enc = layers.embed_sequence(X, vocab_size=vocab_dim, embed_dim=embedding_dim)
	outputs_enc = layers.embed_sequence(Y, vocab_size=vocab_dim, embed_dim=embedding_dim)
	cell_enc = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, inputs=inputs_enc, sequence_length=X_len, dtype=tf.float32, scope='g1')
	# attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=latent_dim, memory=outputs_enc, memory_sequence_length=X_len)
	cell_dec = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim, state_is_tuple=False) # state_is_tuple=False
	# cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attention_mechanism, attention_layer_size=latent_dim)
	# cell_dec_out = tf.contrib.rnn.OutputProjectionWrapper(cell_dec_attn, vocab_dim)
	helper_train = tf.contrib.seq2seq.TrainingHelper(outputs_enc, Y_len)
	init = tf.concat([state_enc.h, state_enc.c], axis=-1)
	# init = cell_dec_attn.zero_state(dtype=tf.float32, batch_size=batch_size)
	projection_layer = layers_core.Dense(vocab_dim, use_bias=False)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_dec, helper=helper_train, initial_state=init, output_layer=projection_layer) # cell=cell_dec_attn
	outputs_dec, last_state, last_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, impute_finished=True, maximum_iterations=max_sent_len)
	loss = tf.contrib.seq2seq.sequence_loss(logits=outputs_dec.rnn_output, targets=Y, weights=Y_mask)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)


	#################### main logic ####################
	if mode == 'train':
		lengths = []
		words_cnt = {}

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(1, 1 + total_epoch):
				for batch in range(data_size // batch_size):
					feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size], Y_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y_mask:data_mask[batch*batch_size:(batch+1)*batch_size]}
					_, now_loss, test = sess.run([train, loss, inputs_enc], feed_dict=feed_dict)
					if batch % 1 == 0:
						print_data = []
						print_data.append(['loss', now_loss])
						print_data.append([])
						for i in range(5):
							test_idx = random.randrange(data_size)
							ret = sess.run(outputs_dec, feed_dict={X:data_x[test_idx:test_idx+1], X_len:data_x_len[test_idx:test_idx+1], Y:data_y[test_idx:test_idx+1], Y_len:data_x_len[test_idx:test_idx+1], Y_mask:data_mask[test_idx:test_idx+1]})
							print_data.append(['original', ' '.join([idx2word[idx] for idx in data_y[test_idx]])])
							print_data.append(['predict', ' '.join([idx2word[idx] for idx in ret.sample_id[0]])])
							if i != 4: print_data.append([])
						print('\n')
						print_table(print_data, title='%d epoch / %d batch' % (epoch, batch + 1), min_width=os.get_terminal_size()[0] - 1)
						print()

			### get statistics
			for batch in range(data_size // batch_size):
				feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size], Y_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y_mask:data_mask[batch*batch_size:(batch+1)*batch_size]}
				ret = sess.run(outputs_dec, feed_dict=feed_dict)
				for rs in ret.sample_id:
					words = [idx2word[idx] for idx in rs]
					cnt = 0
					for word in words:
						if word in ['.', '!', '?']:
							break
						cnt += 1
						if not word in words_cnt:
							words_cnt[word] = 0
						words_cnt[word] += 1
					lengths.append(cnt)

			with open('seq2seq2/out/sent_lengths.txt', 'w', encoding='utf-8') as fp:
				for length in pgbar(lengths, pre='[sent_lengths.txt]'):
					fp.write('%s\n' % length)

			words_sorted = sorted(words_cnt, key=lambda x:words_cnt[x], reverse=True)
			with open('seq2seq2/out/words_sorted.txt', 'w', encoding='utf-8') as fp:
				for word in pgbar(words_sorted, pre='[words_sorted]'):
					fp.write('%s %d\n' % (word, words_cnt[word]))

	elif mode == 'test':
		print('test? comming soon...')

	elif mode == 'visualize':
		#################### load origin and predict data ####################
		### load origin data
		sent_lengths_origin = []
		words_sorted_origin = []
		words_cnt_origin = {}

		with open('seq2seq2/data/train.data', 'r', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[train.data]'):
				temp = json.loads(line)
				words = temp['text'].split()
				sent_lengths_origin.append(len(words))
				for word in words:
					if word in ['.', '!', '?']:
						continue
					if not word in words_cnt_origin:
						words_cnt_origin[word] = 0
					words_cnt_origin[word] += 1
		words_sorted_origin = sorted(words_cnt_origin, key=lambda x:words_cnt_origin[x], reverse=True)
		
		sent_lengths_predict = []
		words_sorted_predict = []
		words_cnt_predict = {}

		### load predict data
		with open('seq2seq2/out/sent_lengths.txt', 'r', encoding='utf-8') as fp:
			lengths = fp.read().strip().split('\n')
			for length in pgbar(lengths, pre='[sent_lengths.txt]'):
				sent_lengths_predict.append(int(length))

		with open('seq2seq2/out/words_sorted.txt', 'r', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in lines:
				word, cnt = line.split()
				cnt = int(cnt)
				words_sorted_predict.append(word)
				words_cnt_predict[word] = cnt

		print('mean sent len: %.1f' % (sum(sent_lengths_predict) / len(sent_lengths_predict)))
		print('total word cnt: %d' % len(words_cnt_predict))

		#################### draw lengths of sentences ####################
		### draw origin data
		x = [i for i in range(100)]
		y = [0 for i in range(100)]
		for length in sent_lengths_origin:
			y[length] += 1
		plt.bar(x, y, 0.5, color=[1, 0, 0], alpha=0.5, label='origin')

		### draw predict data
		x = [i for i in range(100)]
		y = [0 for i in range(100)]
		for length in sent_lengths_predict:
			y[length] += 1
		plt.bar(x, y, 0.5, color=[0, 0, 1], alpha=0.5, label='predict')
		plt.legend(loc='upper right')
		plt.show()
		plt.clf()

		#################### draw words by cnt ####################
		### draw origin data
		x = [i for i in range(100)]
		y = [words_cnt_origin[word] for word in words_sorted_origin[:100]]
		plt.bar(x, y, 0.5, color=[1, 0, 0], alpha=0.5, label='origin')

		### draw predict data
		x = [i for i in range(100)]
		y = [words_cnt_predict[word] if word in words_cnt_predict else 0 for word in words_sorted_origin[:100]] # since we want to diff between original and predict
		plt.bar(x, y, 0.5, color=[0, 0, 1], alpha=0.5, label='predict')

		plt.xticks(x, words_sorted_origin[:100])
		plt.legend(loc='upper right')
		plt.show()
		plt.clf()