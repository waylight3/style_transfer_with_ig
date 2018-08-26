import tensorflow as tf
import random, json, sys, os
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

	print('[PREPRO] setting: train_ratio(%.2f) / dev_ratio(%.2f) / test_ratio(%.2f)' % (train_ratio, dev_ratio, 1.0 - train_ratio - dev_ratio))

	### read data line by line
	print('[PREPRO] loading yelp data start')
	with open('data/yelp/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as fp:
		for d in pgbar(fp.read().strip().split('\n'), pre='[PREPRO]'):
			data.append(json.loads(d))
	print('[PREPRO] loading yelp data finish')

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
	# save train data
	with open('train.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_train, pre='[PREPRO]'):
			fp.write(json.dumps(d) + '\n')
	# save dev data
	with open('dev.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_dev, pre='[PREPRO]'):
			fp.write(json.dumps(d) + '\n')
	# save test data
	with open('test.data', 'w', encoding='utf-8') as fp:
		for d in pgbar(data_test, pre='[PREPRO]'):
			fp.write(json.dumps(d) + '\n')
	print('[PREPRO] saving data start')
