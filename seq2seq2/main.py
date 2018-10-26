import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib import layers
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import argparse, random, random, json, sys, os
from util.util import *
import win_unicode_console

win_unicode_console.enable()
tf.set_random_seed(2015147554)

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize']) # train
	parser.add_argument('--latent_dim', type=int, default=50) # 50
	parser.add_argument('--embedding_dim', type=int, default=100) # 100
	parser.add_argument('--max_sent_len', type=int, default=20, help='maximum number of words in each sentence') # 20
	parser.add_argument('--batch_size', type=int, default=50, help='size of each batch. prefer to be a factor of data size') # 50
	parser.add_argument('--learning_rate', type=float, default=0.003) # 0.003
	parser.add_argument('--total_epoch', type=int, default=20) # 20
	parser.add_argument('--use_word2vec', type=str, default='false', choices=['true', 'false']) # false
	parser.add_argument('--vismode', type=str, default='node_ig_list', choices=['ig', 'sent_len', 'word_cnt', 'sent_ig_list', 'node_ig_list']) # sent_ig_list
	args = parser.parse_args()

	### constants
	UNKNOWN = 0
	GO = 1
	END = 2
	PAD = 3

	### setting arguments
	mode = args.mode
	word2vec = {}
	word2idx = {}
	idx2word = {UNKNOWN:'_?_', GO:'_GO_', END:'_END_', PAD: '_PAD_'}
	word2vec_dim = 0
	word_list = []
	vocab_dim = 4
	latent_dim = args.latent_dim
	max_sent_len = args.max_sent_len
	batch_size = args.batch_size
	learning_rate = args.learning_rate
	total_epoch = args.total_epoch
	use_word2vec = True if args.use_word2vec == 'true' else False
	vismode = args.vismode

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
	data_star = []
	data_size = 0

	### read data
	with open('seq2seq2/data/train.data', encoding='utf-8') as fp:
		lines = fp.read().strip().split('\n')
		for line in pgbar(lines, pre='[train.data]'):
			sent_idx = []
			sent_mask = []
			temp = json.loads(line)
			words = temp['text'].split()
			star = temp['stars']
			for word in words:
				if word in word2idx:
					sent_idx.append(word2idx[word])
					sent_mask.append(1.0)
				else:
					sent_idx.append(UNKNOWN)
					sent_mask.append(0.0)
				if len(sent_idx) >= max_sent_len:
					break
			# while len(sent_idx) < max_sent_len:
			# 	sent_idx.append(END)
			# 	sent_mask.append(0.0)
			data_x.append([GO] + sent_idx + [PAD for _ in range(max_sent_len - len(sent_idx))])
			data_y.append(sent_idx + [END] + [PAD for _ in range(max_sent_len - len(sent_idx))])
			data_mask.append(sent_mask + [1.0] + [0.0 for _ in range(max_sent_len - len(sent_idx))])
			data_x_len.append(max_sent_len + 1)
			data_star.append([1 if i + 1 == star else 0 for i in range(5)])
			data_size += 1

	dev_x = []
	dev_x_len = []	
	dev_y = []
	dev_mask = []
	dev_star = []
	dev_size = 0

	with open('seq2seq2/data/dev.data', encoding='utf-8') as fp:
		lines = fp.read().strip().split('\n')
		for line in pgbar(lines, pre='[dev.data]'):
			sent_idx = []
			sent_mask = []
			temp = json.loads(line)
			words = temp['text'].split()
			star = temp['stars']
			for word in words:
				if word in word2idx:
					sent_idx.append(word2idx[word])
					sent_mask.append(1.0)
				else:
					sent_idx.append(UNKNOWN)
					sent_mask.append(0.0)
				if len(sent_idx) >= max_sent_len:
					break
			# while len(sent_idx) < max_sent_len:
			# 	sent_idx.append(END)
			# 	sent_mask.append(0.0)
			dev_x.append([GO] + sent_idx + [PAD for _ in range(max_sent_len - len(sent_idx))])
			dev_y.append(sent_idx + [END] + [PAD for _ in range(max_sent_len - len(sent_idx))])
			dev_mask.append(sent_mask + [1.0] + [0.0 for _ in range(max_sent_len - len(sent_idx))])
			dev_x_len.append(max_sent_len + 1)
			dev_star.append([1.0 if i + 1 == star else 0.0 for i in range(5)])
			dev_size += 1

	max_sent_len += 1 # +1 for _star_ and <END>

	#################### history ####################
	history = {
		'dev_loss':[], 'dev_acc':[], 'dev_bleu':[],
		'ig':[]
	}

	#################### model ####################
	tf.reset_default_graph()

	X = tf.placeholder(tf.int32, [None, max_sent_len])
	X_len = tf.placeholder(tf.int32, [None])
	Y = tf.placeholder(tf.int32, [None, max_sent_len])
	Y_len = tf.placeholder(tf.int32, [None])
	Y_mask = tf.placeholder(tf.float32, [None, max_sent_len])
	Star = tf.placeholder(tf.float32, [None, 5])

	inputs_enc = layers.embed_sequence(X, vocab_size=vocab_dim, embed_dim=embedding_dim)
	outputs_enc = layers.embed_sequence(Y, vocab_size=vocab_dim, embed_dim=embedding_dim)
	cell_enc = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim)
	outputs_enc, state_enc = tf.nn.dynamic_rnn(cell=cell_enc, inputs=inputs_enc, sequence_length=X_len, dtype=tf.float32, scope='g1')
	cell_dec = tf.contrib.rnn.BasicLSTMCell(num_units=latent_dim, state_is_tuple=False)
	helper_train = tf.contrib.seq2seq.TrainingHelper(outputs_enc, Y_len)
	# init = tf.concat([state_enc.h, state_enc.c], axis=-1)
	# g1 = tf.concat([state_enc.h, state_enc.c], axis=-1)
	# g2 = tf.multiply(state_enc.h, state_enc.c)
	# gg = tf.concat([g1, g2], axis=-1)
	g1 = tf.concat([state_enc.h, Star], axis=-1)
	latent = tf.layers.dense(g1, latent_dim)
	init = tf.layers.dense(latent, latent_dim * 2)
	projection_layer = layers_core.Dense(vocab_dim, use_bias=False)
	decoder = tf.contrib.seq2seq.BasicDecoder(cell=cell_dec, helper=helper_train, initial_state=init, output_layer=projection_layer)
	outputs_dec, last_state, last_seq_len = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, impute_finished=True, maximum_iterations=max_sent_len)
	loss = tf.contrib.seq2seq.sequence_loss(logits=outputs_dec.rnn_output, targets=Y, weights=Y_mask)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train = optimizer.minimize(loss)

	saver = tf.train.Saver(max_to_keep=total_epoch)


	#################### main logic ####################
	if mode == 'train':
		lengths = []
		words_cnt = {}

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(1, 1 + total_epoch):
				now_loss, now_acc, now_bleu, batch = 0, 0, 0, 0
				for batch in pgbar(range(data_size // batch_size), pre='[%d epoch]' % epoch):
					feed_dict={X:data_x[batch*batch_size:(batch+1)*batch_size], X_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y:data_y[batch*batch_size:(batch+1)*batch_size], Y_len:data_x_len[batch*batch_size:(batch+1)*batch_size], Y_mask:data_mask[batch*batch_size:(batch+1)*batch_size], Star:data_star[batch*batch_size:(batch+1)*batch_size]}
					_, now_loss, ret, test = sess.run([train, loss, outputs_dec, inputs_enc], feed_dict=feed_dict)
					if (batch + 1) % 20 == 0:
						### get accuracy and bleu score
						now_acc = seq2seq_accuracy(logits=ret.sample_id, targets=data_y[batch*batch_size:(batch+1)*batch_size], weights=data_mask[batch*batch_size:(batch+1)*batch_size])
						now_bleu = seq2seq_bleu(logits=ret.sample_id, targets=data_y[batch*batch_size:(batch+1)*batch_size])

						### print info
						test_idx = random.randrange(data_size)
						ret = sess.run(outputs_dec, feed_dict={X:data_x[test_idx:test_idx+1], X_len:data_x_len[test_idx:test_idx+1], Y:data_y[test_idx:test_idx+1], Y_len:data_x_len[test_idx:test_idx+1], Y_mask:data_mask[test_idx:test_idx+1], Star:data_star[test_idx:test_idx+1]})
						if batch != data_size // batch_size - 1: print()
						print('loss: %.4f / acc: %.4f / bleu: %.4f' % (now_loss, now_acc, now_bleu))
						print('original: %s' % ' '.join([idx2word[idx] for idx in data_y[test_idx]]))
						print('predict : %s' % ' '.join([idx2word[idx] for idx in ret.sample_id[0]]))
				
				saver.save(sess, 'seq2seq2/save/model_%d' % epoch)

				### get IG info
				t_grads = tf.gradients(outputs_dec, inputs_enc)
				grads, ie = sess.run([t_grads, inputs_enc], feed_dict={X:interpolate([0 for i in range(max_sent_len)], data_x[1], 100), X_len:[data_x_len[1]] * 100, Y:[data_y[1]] * 100, Y_len:[data_x_len[1]] * 100, Y_mask:[data_mask[1]] * 100, Star:[data_star[1]] * 100})
				grads = np.array(grads)
				ie = np.array(ie[0]) # select [0] since we calc 100 data for interpolation
				agrads = np.average(grads, axis=1)[0]
				ig = []
				for i in range(max_sent_len):
					t = 0.0
					for j in range(embedding_dim):
						t += ie[i][j] * agrads[i][j]
					ig.append(t)
				history['ig'].append(ig)

				### print validation info
				dev_loss = 0.0
				dev_acc = 0.0
				dev_bleu = 0.0
				for batch in range(dev_size // batch_size):
					feed_dict={X:dev_x[batch*batch_size:(batch+1)*batch_size], X_len:dev_x_len[batch*batch_size:(batch+1)*batch_size], Y:dev_y[batch*batch_size:(batch+1)*batch_size], Y_len:dev_x_len[batch*batch_size:(batch+1)*batch_size], Y_mask:dev_mask[batch*batch_size:(batch+1)*batch_size], Star:dev_star[batch*batch_size:(batch+1)*batch_size]}
					now_loss, ret = sess.run([loss, outputs_dec], feed_dict=feed_dict)

					### get accuracy and bleu score
					now_acc = seq2seq_accuracy(logits=ret.sample_id, targets=data_y[batch*batch_size:(batch+1)*batch_size], weights=data_mask[batch*batch_size:(batch+1)*batch_size])
					now_bleu = seq2seq_bleu(logits=ret.sample_id, targets=data_y[batch*batch_size:(batch+1)*batch_size])
					dev_loss += now_loss
					dev_acc += now_acc
					dev_bleu += now_bleu
				dev_loss /= dev_size // batch_size
				dev_acc /= dev_size // batch_size
				dev_bleu /= dev_size // batch_size
				history['dev_loss'].append(dev_loss)
				history['dev_acc'].append(dev_acc)
				history['dev_bleu'].append(dev_bleu)
				
				### print info
				print()
				print('dev loss: %.6f' % dev_loss)
				print('dev acc : %.6f' % dev_acc)
				print('dev bleu: %.6f' % dev_bleu)
				print()

			### print history
			print_data = []
			print_data.append(['epoch', 'dev loss', 'dev acc', 'dev bleu'])
			for epoch in range(total_epoch):
				print_data.append(['%d' % (epoch + 1), '%.4f' % history['dev_loss'][epoch], '%.4f' % history['dev_acc'][epoch], '%.4f' % history['dev_bleu'][epoch]])
			print_table(print_data, title='history')

			### save history
			with open('seq2seq2/out/history_dev_loss.txt', 'w', encoding='utf-8') as fp:
				for i in pgbar(range(len(history['dev_loss'])), pre='[history_dev_loss.txt]'):
					fp.write('%s\n' % history['dev_loss'][i])
			with open('seq2seq2/out/history_dev_acc.txt', 'w', encoding='utf-8') as fp:
				for i in pgbar(range(len(history['dev_acc'])), pre='[history_dev_acc.txt]'):
					fp.write('%s\n' % history['dev_acc'][i])
			with open('seq2seq2/out/history_dev_bleu.txt', 'w', encoding='utf-8') as fp:
				for i in pgbar(range(len(history['dev_bleu'])), pre='[history_dev_bleu.txt]'):
					fp.write('%s\n' % history['dev_bleu'][i])
			with open('seq2seq2/out/history_ig.txt', 'w', encoding='utf-8') as fp:
				for i in pgbar(range(len(history['ig'])), pre='[history_ig.txt]'):
					fp.write('%s\n' % history['ig'][i])

			### get statistics
			for batch in range(dev_size // batch_size):
				feed_dict={X:dev_x[batch*batch_size:(batch+1)*batch_size], X_len:dev_x_len[batch*batch_size:(batch+1)*batch_size], Y:dev_y[batch*batch_size:(batch+1)*batch_size], Y_len:dev_x_len[batch*batch_size:(batch+1)*batch_size], Y_mask:dev_mask[batch*batch_size:(batch+1)*batch_size], Star:dev_star[batch*batch_size:(batch+1)*batch_size]}
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

			### save statistics
			with open('seq2seq2/out/sent_lengths.txt', 'w', encoding='utf-8') as fp:
				for length in pgbar(lengths, pre='[sent_lengths.txt]'):
					fp.write('%s\n' % length)
			words_sorted = sorted(words_cnt, key=lambda x:words_cnt[x], reverse=True)
			with open('seq2seq2/out/words_sorted.txt', 'w', encoding='utf-8') as fp:
				for word in pgbar(words_sorted, pre='[words_sorted]'):
					fp.write('%s %d\n' % (word, words_cnt[word]))

	elif mode == 'test':
		test_x = []
		test_x_len = []	
		test_y = []
		test_mask = []
		test_star = []
		test_size = 0

		max_sent_len -= 1
		with open('seq2seq2/data/test.data', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[test.data]'):
				sent_idx = []
				sent_mask = []
				temp = json.loads(line)
				words = temp['text'].split()
				star = temp['stars']
				for word in words:
					if word in word2idx:
						sent_idx.append(word2idx[word])
						sent_mask.append(1.0)
					else:
						sent_idx.append(UNKNOWN)
						sent_mask.append(0.0)
					if len(sent_idx) >= max_sent_len:
						break
				# while len(sent_idx) < max_sent_len:
				# 	sent_idx.append(END)
				# 	sent_mask.append(0.0)
				test_x.append([GO] + sent_idx + [PAD for _ in range(max_sent_len - len(sent_idx))])
				test_y.append(sent_idx + [END] + [PAD for _ in range(max_sent_len - len(sent_idx))])
				test_mask.append(sent_mask + [1.0] + [0.0 for _ in range(max_sent_len - len(sent_idx))])
				test_x_len.append(max_sent_len + 1)
				test_star.append([1.0 if i + 1 == star else 0.0 for i in range(5)])
				test_size += 1

		max_sent_len += 1
		test_size = 100

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, 'seq2seq2/save/model_%d' % args.total_epoch)

			ig_list = []
			sent_gen_list = []
			node_ig1_list=[]
			node_ig2_list=[]
			node_ig3_list=[]

			for data_index in pgbar(range(test_size), pre='[test]'):
				#################### get ig of input ####################
				feed_dict={X:test_x[data_index:data_index+1], X_len:test_x_len[data_index:data_index+1], Y:test_y[data_index:data_index+1], Y_len:test_x_len[data_index:data_index+1], Y_mask:test_mask[data_index:data_index+1], Star:test_star[data_index:data_index+1]}
				ret = sess.run(outputs_dec, feed_dict=feed_dict)

				sent_gen = ' '.join([idx2word[idx] for idx in ret.sample_id[0]])

				### get IG info
				t_grads = tf.gradients(outputs_dec, inputs_enc)
				grads, ie = sess.run([t_grads, inputs_enc], feed_dict={X:interpolate([0 for i in range(max_sent_len)], test_x[data_index], 100), X_len:[test_x_len[data_index]] * 100, Y:[test_y[data_index]] * 100, Y_len:[test_x_len[data_index]] * 100, Y_mask:[test_mask[data_index]] * 100, Star:[test_star[data_index]] * 100})
				grads = np.array(grads)
				ie = np.array(ie[0]) # select [0] since we calc 100 data for interpolation
				agrads = np.average(grads, axis=1)[0]
				ig = []
				for i in range(max_sent_len):
					t = 0.0
					for j in range(embedding_dim):
						t += ie[i][j] * agrads[i][j]
					ig.append(t)

				ig_list.append(ig)
				sent_gen_list.append(sent_gen)

				#################### get IG of g1/g2/latent ####################
				### get IG of g1
				t_grads = tf.gradients(outputs_dec, g1)
				grads, _g1 = sess.run([t_grads, g1], feed_dict={X:interpolate([0 for i in range(max_sent_len)], test_x[data_index], 100), X_len:[test_x_len[data_index]] * 100, Y:[test_y[data_index]] * 100, Y_len:[test_x_len[data_index]] * 100, Y_mask:[test_mask[data_index]] * 100, Star:[test_star[data_index]] * 100})
				grads = np.array(grads)
				_g1 = np.array(_g1[0]) # select [0] since we calc 100 data for interpolation
				agrads = np.average(grads, axis=1)[0]

				ig = []
				for i in range(embedding_dim // 2 + 5):
					t = _g1[i] * agrads[i]
					ig.append(t)

				node_ig1_list.append(ig)

				### get IG of g2
				# t_grads = tf.gradients(outputs_dec, g2)
				# grads, _g2 = sess.run([t_grads, g2], feed_dict={X:interpolate([0 for i in range(max_sent_len)], data_x[data_index], 100), X_len:[data_x_len[data_index]] * 100, Y:[data_y[data_index]] * 100, Y_len:[data_x_len[data_index]] * 100, Y_mask:[data_mask[data_index]] * 100})
				# grads = np.array(grads)
				# _g2 = np.array(_g2[0]) # select [0] since we calc 100 data for interpolation
				# agrads = np.average(grads, axis=1)[0]

				# ig = []
				# for i in range(embedding_dim // 2):
				# 	t = _g2[i] * agrads[i]
				# 	ig.append(t)

				# node_ig2_list.append(ig)

				### get IG of latent
				t_grads = tf.gradients(outputs_dec, latent)
				grads, _g3 = sess.run([t_grads, latent], feed_dict={X:interpolate([0 for i in range(max_sent_len)], test_x[data_index], 100), X_len:[test_x_len[data_index]] * 100, Y:[test_y[data_index]] * 100, Y_len:[test_x_len[data_index]] * 100, Y_mask:[test_mask[data_index]] * 100, Star:[test_star[data_index]] * 100})
				grads = np.array(grads)
				_g3 = np.array(_g3[0]) # select [0] since we calc 100 data for interpolation
				agrads = np.average(grads, axis=1)[0]

				ig = []
				for i in range(latent_dim):
					t = _g3[i] * agrads[i]
					ig.append(t)

				node_ig3_list.append(ig)

		### save test info
		with open('seq2seq2/out/sent_gen_list.txt', 'w', encoding='utf-8') as fp:
			for sg in pgbar(sent_gen_list, pre='[sent_gen_list.txt]'):
				fp.write('%s\n' % sg)

		with open('seq2seq2/out/ig_list.txt', 'w', encoding='utf-8') as fp:
			for ig in pgbar(ig_list, pre='[ig_list.txt]'):
				fp.write(' '.join(list(map(lambda x: '%.6f' % x, ig))) + '\n')

		with open('seq2seq2/out/node_ig1_list.txt', 'w', encoding='utf-8') as fp:
			for ig in pgbar(node_ig1_list, pre='[node_ig1_list.txt]'):
				fp.write(' '.join(list(map(lambda x: '%.6f' % x, ig))) + '\n')

		# with open('seq2seq2/out/node_ig2_list.txt', 'w', encoding='utf-8') as fp:
		# 	for ig in pgbar(node_ig2_list, pre='[node_ig2_list.txt]'):
		# 		fp.write(' '.join(list(map(lambda x: '%.6f' % x, ig))) + '\n')

		with open('seq2seq2/out/node_ig3_list.txt', 'w', encoding='utf-8') as fp:
			for ig in pgbar(node_ig3_list, pre='[node_ig3_list.txt]'):
				fp.write(' '.join(list(map(lambda x: '%.6f' % x, ig))) + '\n')

	elif mode == 'visualize':
		#################### load origin and predict data ####################
		### load origin data
		sent_lengths_origin = []
		words_sorted_origin = []
		words_cnt_origin = {}

		train_text_1 = 0
		train_star_1 = 0

		with open('seq2seq2/data/train.data', 'r', encoding='utf-8') as fp:
			lines = fp.read().strip().split('\n')
			for line in pgbar(lines, pre='[train.data]'):
				temp = json.loads(line)
				words = temp['text'].split()
				if train_text_1 == 0:
					train_text_1 = 1
				elif train_text_1 == 1:
					train_text_1 = words
					train_star_1 = temp['stars']
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
			for line in pgbar(lines, pre='[words_sorted.txt]'):
				word, cnt = line.split()
				cnt = int(cnt)
				words_sorted_predict.append(word)
				words_cnt_predict[word] = cnt

		print('mean sent len: %.1f' % (sum(sent_lengths_predict) / len(sent_lengths_predict)))
		print('total word cnt: %d' % len(words_cnt_predict))

		if vismode == 'ig':
			ig_list = []
			with open('seq2seq2/out/history_ig.txt', 'r', encoding='utf-8') as fp:
				lines = fp.read().strip().split('\n')
				for line in pgbar(lines, pre='[history_ig.txt]'):
					ig = eval(line)
					ig_list.append(ig)

			x_ticks = [str(train_star_1)] + train_text_1 + ['<END>' for i in range(20)]
			fig, ax = plt.subplots()
			ax.imshow(ig_list, cmap='YlGn', aspect=1)
			ax.set_xticks([i for i in range(21)])
			ax.set_xticklabels(x_ticks[:21])
			ax.set_yticks([i for i in range(total_epoch)])
			ax.set_yticklabels([str(i + 1) for i in range(total_epoch)])
			plt.show()
			plt.clf()

		#################### draw lengths of sentences ####################
		if vismode == 'sent_len':
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
		if vismode == 'word_cnt':
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

		#################### draw ig values of each sentence ####################
		if vismode == 'sent_ig_list':
			test_data = []
			ig_list = []
			star_list = []
			data_size = 20
			# x_len = []

			### load data
			with open('seq2seq2/data/test.data', 'r', encoding='utf-8') as fp:
				lines = fp.read().strip().split('\n')
				for line in pgbar(lines, pre='[test.data]'):
					temp = json.loads(line)
					words = temp['text'].split()
					if len(words) < 20:
						words += [' '] * (20 - len(words))
					# piv_end = len(words)
					# for i, word in enumerate(words):
					# 	if word in ['.', '!', '?']:
					# 		piv_end = i + 1
					# 		break
					# x_len.append(piv_end)
					test_data.append(words)
					star_list.append(temp['stars'])
			test_data = np.array(test_data[:data_size])
			star_list = np.array(star_list[:data_size])

			with open('seq2seq2/out/ig_list.txt', 'r', encoding='utf-8') as fp:
				lines = fp.read().strip().split('\n')
				for line in pgbar(lines, pre='[ig_list.txt]'):
					temp = list(map(float, line.split()))[1:]
					# piv_end = x_len.pop(0)
					# ig_list.append(rescale(temp[:piv_end]) + [0] * (len(temp) - piv_end))
					ig_list.append(rescale(temp))
			ig_list = np.array(ig_list[:data_size])

			### draw ig values
			elev_min = ig_list.min()
			elev_max = ig_list.max()
			mid_val = 0
			x = [i for i in range(len(ig_list[0]))]
			fig, ax = plt.subplots()
			fig.set_size_inches(30, data_size * 3 // 5)
			im, cbar = heatmap(ig_list, star_list, x, ax=ax, cmap='seismic', cbarlabel='contribution', cbar_kw={'fraction':0.1}, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
			annotate_heatmap(im, texts=test_data, threshold=-0.5, threshold2=0.3)
			ax.set_aspect(0.5)
			fig.tight_layout()
			plt.savefig('seq2seq2/out/sent_ig_list.png')
			# plt.show()

		if vismode == 'node_ig_list':
			node_ig1_list = []
			node_ig2_list = []
			node_ig3_list = []
			with open('seq2seq2/out/node_ig1_list.txt', 'r', encoding='utf-8') as fp:
				lines = fp.read().strip().split('\n')
				for line in pgbar(lines, pre='[node_ig1_list.txt]'):
					temp = list(map(float, line.split()))
					node_ig1_list.append(temp)
			node_ig1_list = np.array(node_ig1_list)

			# with open('seq2seq2/out/node_ig2_list.txt', 'r', encoding='utf-8') as fp:
			# 	lines = fp.read().strip().split('\n')
			# 	for line in pgbar(lines, pre='[node_ig2_list.txt]'):
			# 		temp = list(map(float, line.split()))
			# 		node_ig2_list.append(temp)
			# node_ig2_list = np.array(node_ig2_list)

			with open('seq2seq2/out/node_ig3_list.txt', 'r', encoding='utf-8') as fp:
				lines = fp.read().strip().split('\n')
				for line in pgbar(lines, pre='[node_ig3_list.txt]'):
					temp = list(map(float, line.split()))
					node_ig3_list.append(temp)
			node_ig3_list = np.array(node_ig3_list)

			elev_min = node_ig1_list.min()
			elev_max = node_ig1_list.max()
			mid_val = 0
			x_ticks = ['%d' % i for i in range(embedding_dim // 2)]
			fig, ax = plt.subplots()
			im = ax.imshow(node_ig1_list, cmap='seismic', aspect=1, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
			ax.set_xticks([i for i in range(embedding_dim // 2 + 5)])
			ax.set_xticklabels(x_ticks + ['1', '2', '3', '4', '5'])
			ax.set_yticks([i for i in range(100)])
			ax.set_yticklabels(['%d' % i for i in range(100)])
			cbar = ax.figure.colorbar(im, ax=ax)
			cbar.ax.set_ylabel('contribution', rotation=-90, va="bottom")
			fig.tight_layout()
			plt.show()
			plt.clf()

			# x_ticks = ['%d' % i for i in range(embedding_dim // 2)]
			# fig, ax = plt.subplots()
			# ax.imshow(node_ig2_list, cmap='YlGn', aspect=1)
			# ax.set_xticks([i for i in range(50)])
			# ax.set_xticklabels(x_ticks)
			# ax.set_yticks([i for i in range(100)])
			# ax.set_yticklabels(['%d' % i for i in range(100)])
			# plt.show()
			# plt.clf()

			elev_min = node_ig3_list.min()
			elev_max = node_ig3_list.max()
			mid_val = 0
			x_ticks = ['%d' % i for i in range(latent_dim)]
			fig, ax = plt.subplots()
			im = ax.imshow(node_ig3_list, cmap='seismic', aspect=1, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
			ax.set_xticks([i for i in range(latent_dim)])
			ax.set_xticklabels(x_ticks)
			ax.set_yticks([i for i in range(100)])
			ax.set_yticklabels(['%d' % i for i in range(100)])
			cbar = ax.figure.colorbar(im, ax=ax)
			cbar.ax.set_ylabel('contribution', rotation=-90, va="bottom")
			fig.tight_layout()
			plt.show()
			plt.clf()