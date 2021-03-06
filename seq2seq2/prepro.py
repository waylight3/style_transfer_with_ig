import tensorflow as tf
import argparse, random, json, sys, os
from nltk.tokenize import sent_tokenize, word_tokenize
from util.util import pgbar

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--total_data_size', type=int, default=100000, help='number of data to prepro (use all if total_data_size = 0)') # 10000
	parser.add_argument('--return_data_size', type=int, default=100000, help='number of data to return (use all if total_data_size = 0)') # 10000
	parser.add_argument('--train_ratio', type=float, default=0.7)
	parser.add_argument('--dev_ratio', type=float, default=0.2)
	parser.add_argument('--max_word', type=int, default=20000, help='maximum total number of words') # 10000
	parser.add_argument('--max_sent_len', type=int, default=20, help='maximum number of words in each sentence')
	parser.add_argument('--min_sent_len', type=int, default=5, help='minimum number of words in each sentence')
	parser.add_argument('--memory_saving_mode', type=str, default='false', choices=['true', 'false'])

	args = parser.parse_args()

	### data split ratio for cross validation
	total_data_size = args.total_data_size
	return_data_size = args.return_data_size
	train_ratio = args.train_ratio
	dev_ratio = args.dev_ratio
	max_sent_len = args.max_sent_len
	min_sent_len = args.min_sent_len
	memory_saving_mode = True if args.memory_saving_mode == 'true' else False
	
	### data list for splitted data
	data = []
	data_train = []
	data_dev = []
	data_test = []

	### for word2vec dict
	max_word = args.max_word
	word_dict = {}
	word_cnt = {}

	#################### load data ####################
	print('[PREPRO] setting: total_data_size(%d) / return_data_size(%d) / train_ratio(%.2f) / dev_ratio(%.2f) / test_ratio(%.2f)' % (total_data_size, return_data_size, train_ratio, dev_ratio, 1.0 - train_ratio - dev_ratio))

	### read data line by line
	print('[PREPRO] loading yelp data start')
	with open('data/yelp/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as fp:
		if memory_saving_mode:
			while True:
				d = fp.readline()
				if not d: break
				data.append(json.loads(d))
				if len(data) >= total_data_size: break
		else:
			for d in pgbar(fp.read().strip().split('\n'), pre='[yelp_academic_dataset_review.json]'):
				data.append(json.loads(d))
	print('[PREPRO] loading yelp data finish (total %d lines)' % len(data))

	# shuffle data to remove order-bias
	random.shuffle(data)
	if total_data_size > 0:
		data = data[:total_data_size]


	#################### prepro word2vec ####################
	print('[PREPRO] setting: max_word(%d)' % max_word)

	### split into sentences/words and count all words
	print('[PREPRO] counting words start')
	for d in pgbar(data, pre='[data]'):
		sents = sent_tokenize(d['text'].strip())
		for sent in sents:
			sent = sent.lower()
			words = word_tokenize(sent.strip())
			# skip if sentence is too long or short
			if len(words) > max_sent_len or len(words) < min_sent_len: continue
			for word in words:
				if not word in word_cnt:
					word_cnt[word] = 0
				word_cnt[word] += 1
	print('[PREPRO] counting words finish')

	### count words and find top words
	total_word = len(word_cnt)
	print('[PREPRO] total %d words find' % total_word)
	word_list = sorted(word_cnt, key=lambda w:word_cnt[w], reverse=True)[:max_word]
	word_set = set(word_list)
	with open('seq2seq2/data/word_list.data', 'w', encoding='utf-8') as fp:
		for word in pgbar(word_list, pre='[word_list.data]'):
			fp.write(word + '\n')

	word2vec_result = []

	### read word2vec from glove
	print('[PREPRO] matching used words and word2vec start')
	with open('data/glove/glove.6B.100d.txt', 'r', encoding='utf-8') as fp:
		word2vec_list = fp.read().strip().split('\n')
		for word2vec in pgbar(word2vec_list, pre='[glove.6B.100d.txt]'):
			word = word2vec.split()[0]
			# use only used words
			if not word in word_set: continue
			word2vec_result.append(word2vec)
	print('[PREPRO] matching used words and word2vec finish')

	### save word2vec data
	print('[PREPRO] saving word2vec start')
	with open('seq2seq2/data/word2vec.data', 'w', encoding='utf-8') as fp:
		for wv in pgbar(word2vec_result, pre='[word2vec.data]'):
			fp.write(wv + '\n')
	print('[PREPRO] saving word2vec finish')


	#################### prepro data ####################
	print('[PREPRO] splitting data start')
	data_ret = []
	for d in pgbar(data, pre='[data split]'):
		sents = sent_tokenize(d['text'].strip())
		for sent in sents:
			sent = sent.lower()
			words = word_tokenize(sent.strip())
			# skip if sentence is too long
			if len(words) > max_sent_len or len(words) < min_sent_len: continue
			data_ret.append({'text':' '.join(words), 'stars':d['stars']})
	random.shuffle(data_ret)
	data_ret = data_ret[:return_data_size]
	print('[PREPRO] total %d sentences catched' % len(data_ret))

	# split into train/dev/test
	data_size = len(data_ret)
	data_train = data_ret[:int(data_size * train_ratio)]
	data_dev = data_ret[int(data_size * train_ratio):int(data_size * (train_ratio + dev_ratio))]
	data_test = data_ret[int(data_size * (train_ratio + dev_ratio)):]
	print('[PREPRO] splitting data finish')

	### save data
	print('[PREPRO] saving data start')
	# check is data folder exists
	if not os.path.exists('seq2seq2/data'):
		os.makedirs('seq2seq2/data')
	# save train data
	with open('seq2seq2/data/train.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_train, pre='[train.data]'):
			fp.write(json.dumps(d) + '\n')
	# save dev data
	with open('seq2seq2/data/dev.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_dev, pre='[dev.data]'):
			fp.write(json.dumps(d) + '\n')
	# save test data
	with open('seq2seq2/data/test.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_test, pre='[test.data]'):
			fp.write(json.dumps(d) + '\n')
	print('[PREPRO] saving data finish')

	
