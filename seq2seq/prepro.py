import sys
import json

# args : source_file_name target_file_name

out_data = []
idx = 0
# [{"id":idx, "stars":stars, "review":review}]
with open(sys.argv[1], 'r', encoding = 'utf-8') as data_file:
	print('Success Open')
	for line in data_file:
		data = json.loads(line)
		dic = {'id':idx, 'stars':data['stars'], 'review':data['text']}
		out_data.append(dic)
		idx += 1
		if idx % 100 == 0:
			print("Processed {}'th data...".format(idx))
		if idx == 10000:
			print("End json loading")
			break

with open(sys.argv[2], 'w', encoding = 'utf-8') as out_file:
	json.dump(out_data, out_file, ensure_ascii = False)