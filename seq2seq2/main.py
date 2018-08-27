import tensorflow as tf
import argparse, json, sys, os
from util.util import pgbar

if __name__ == '__main__':
	### parsing arguments
	parser = argparse.ArgumentParser(description='This is main file of the seq2seq2 sub project')
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
	args = parser.parse_args()

	#################### model ####################
	

	#################### main logic ####################

	if args.mode == 'train':
		print('train? comming soon...')
	elif args.mode == 'test':
		print('test? comming soon...')
