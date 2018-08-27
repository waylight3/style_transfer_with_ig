import tensorflow as tf
import random, json, sys, os
from nltk.tokenize import sent_tokenize, word_tokenize
from util.util import pgbar

if __name__ == '__main__':
	### data split ratio for cross validation
	train_ratio = 0.7
	dev_ratio = 0.2
	
	### data list for splitted data
	data = []
	data_train = []
	data_dev = []
	data_test = []

	### for word2vec dict
	word_top = 1000
	word_dict = {}
	word_cnt = {}

	#################### prepro data ####################

	print('[PREPRO] setting: train_ratio(%.2f) / dev_ratio(%.2f) / test_ratio(%.2f)' % (train_ratio, dev_ratio, 1.0 - train_ratio - dev_ratio))

	### read data line by line
	print('[PREPRO] loading yelp data start')
	with open('data/yelp/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as fp:
		for d in pgbar(fp.read().strip().split('\n'), pre='[yelp_academic_dataset_review.json]'):
			data.append(json.loads(d))
	print('[PREPRO] loading yelp data finish (total %d lines)' % len(data))

	### split data
	print('[PREPRO] splitting data start')
	# shuffle data to remove order-bias
	random.shuffle(data)
	# split into train/dev/test
	data_size = len(data)
	data_train = data[:int(data_size * train_ratio)]
	data_dev = data[int(data_size * train_ratio):int(data_size * (train_ratio + dev_ratio))]
	data_test = data[int(data_size * (train_ratio + dev_ratio)):]
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

	#################### prepro word2vec ####################

	print('[PREPRO] setting: word_top(%d)' % word_top)

	### split into sentences/words and count all words
	print('[PREPRO] counting words start')
	for d in pgbar(data, pre='[data]'):
		sents = sent_tokenize(d['text'].strip())
		for sent in sents:
			words = word_tokenize(sent.strip())
			for word in words:
				if not word in word_cnt:
					word_cnt[word] = 0
				word_cnt[word] += 1
	print('[PREPRO] counting words finish')

	### count words and find top words
	total_word = len(word_cnt)
	print('[PREPRO] total %d words find' % total_word)
	word_set = set(sorted(word_cnt, key=lambda w:word_cnt[w], reverse=True)[:word_top])

	word2vec_result = []

	### read word2vec from glove
	print('[PREPRO] matching used words and word2vec start')
	with open('data/glove/glove.6B.100d.txt', 'r', encoding='utf-8') as fp:
		word2vec_list = fp.read().strip().split('\n')
		for word2vec in pgbar(word2vec_list, pre='glove.6B.100d.txt'):
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
