import tensorflow as tf
import json, sys, os
from util.util import pgbar

if __name__ == '__main__':
	with open('data/yelp/yelp_academic_dataset_review.json', 'r') as fp:
		for i in range(10):
			data = fp.readline()
			print(data)
