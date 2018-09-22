import sys, os, random
import json
from nltk.tokenize import sent_tokenize, word_tokenize

# args : source_file_name target_file_name

out_data = []
out_data_sent = []

idx = 0
# [{"id":idx, "stars":stars, "review":review}]
with open(sys.argv[1], 'r', encoding = 'utf-8') as data_file:
	print('Success Openning')
	for line in data_file:
		data = json.loads(line)
		dic = {'id':idx, 'stars':data['stars'], 'text':data['text']}
		out_data.append(dic)
		idx += 1
		if idx % 100 == 0:
			print("Processed {}'th data...".format(idx))
		if idx == 10000:
			print("End json loading")
			break

with open(sys.argv[2], 'w', encoding = 'utf-8') as out_file:
	json.dump(out_data, out_file, ensure_ascii = False)

word_to_use = {}
for data in out_data:
	sents = sent_tokenize(data['text'].strip())
	for sent in sents:
		words = word_tokenize(sent.strip())
		for word in words:
			if word not in word_to_use:
				word_to_use[word] = 0
			word_to_use[word] += 1
		out_data_sent.append({'text' : sent, 'stars' : data['stars']})

word_to_use = set(sorted(word_to_use, key=lambda w:word_to_use[w], reverse=True)[:200])

word_to_vec = []

with open('data/glove/glove.6B.100d.txt', 'r', encoding='utf-8') as fp:
	wvs = fp.read().strip().split('\n')
	for wv in wvs:
		word = wv.split()[0]
		if not word in word_set: continue
		word_to_vec.append(wv)

with open('data/word2vec.txt', 'w', encoding='utf-8') as fp:
	for wv in word_to_vec:
		fp.write(wv + '\n')

random.shuffle(out_data_sent)

with open('data/train.txt', 'w', encoding = 'utf-8') as fp:
	for data in out_data_sent[0:100]:
		fp.write(json.dumps(data) + '\n')

with open('data/dev.txt', 'w', encoding = 'utf-8') as fp:
	for data in out_data_sent[100:200]:
		fp.write(json.dumps(data) + '\n')

with open('data/test.txt', 'w', encoding = 'utf-8') as fp:
	for data in out_data_sent[200:300]:
		fp.write(json.dumps(data) + '\n')
