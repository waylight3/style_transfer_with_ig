from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import time, os

def pgbar(data, pre='', post='', bar_icon='=', space_icon=' ', total_display=1000, show_running_time=True, end='\r'):
	'PRE [PGBAR] [000%] [00h00m00s] POST'

	size = len(data)
	tsize, _ = os.get_terminal_size()
	bsize = tsize - len(pre) - len(post) - 12
	if show_running_time:
		bsize -= 12

	time_start = time.time()

	for i, d in enumerate(data):
		bar = bar_icon * (bsize * (i + 1) // size // len(bar_icon))
		space = space_icon * (bsize - len(bar))
		total_time = int(time.time() - time_start)
		hour = total_time // 3600
		minute = total_time % 3600 // 60
		second = total_time % 60
		running_time = ''
		if show_running_time:
			running_time = ' [%02dh%02dm%02ds]' % (hour, minute, second)
		if size < total_display or i % (size // total_display) == 0 or i == size - 1:
			print('%s [%s%s] [%3d%%]%s %s' % (pre, bar, space, 100 * (i + 1) // size, running_time, post), end=end)
		if i == size - 1:
			print('')
		yield d

def print_table(table, title='Title', min_width=0):
	### find error and finish
	if len(table) < 1:
		print('[func::print_table] ERROR: NO ROW DATA')
		return

	if len(table[0]) < 1:
		print('[func::print_table] ERROR: NO COLUMN DATA')
		return

	### get max width of hole table and each col
	max_col = [0 for i in range(len(table[0]))]
	for tr in table:
		for i, td in enumerate(tr):
			now_col = len(str(td)) + 2
			max_col[i] = max(max_col[i], now_col)
	max_width = max(sum(max_col) + len(max_col) + 1, len(title) + 4)
	if max_width < min_width:
		max_col[-1] += min_width - max_width
		max_width = min_width

	### print table
	print('+' + '-' * (max_width - 2) + '+')
	print('|%s%s%s|' % (' ' * ((max_width - len(title) - 2) // 2), title, ' ' * ((max_width - len(title) - 1) // 2)))
	print('+' + '-' * (max_width - 2) + '+')
	for tr in table:
		print('+' + '-' * (max_width - 2) + '+')
		if len(tr) < 1:
			continue
		print('|', end='')
		for i, td in enumerate(tr):
			print(' %s%s' % (td , ' ' * (max_col[i] - len(str(td)) - 1)) + '|', end='')
		print()
	print('+' + '-' * (max_width - 2) + '+')

def seq2seq_accuracy(logits, targets, weights):
	batch_size = len(logits)
	total_acc = 0.0
	for bc in range(batch_size):
		acc = 0.0
		cnt = 0.0
		for i in range(21):
			if weights[bc][i] < 0.01: continue
			cnt += 1.0
			if targets[bc][i] != logits[bc][i]: continue
			acc += 1.0
		acc /= cnt
		total_acc += acc
	total_acc /= batch_size
	return total_acc

def seq2seq_bleu(logits, targets):
	batch_size = len(logits)
	total_bleu = 0.0
	for bc in range(batch_size):
		score = sentence_bleu([targets[bc]], logits[bc], weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
		total_bleu += score
	total_bleu /= batch_size
	return total_bleu

def interpolate(v1, v2, step):
	ret = []
	for i in range(1, step + 1):
		ret.append([v1[j] + (v2[j] - v1[j]) * i / step for j in range(len(v1))])
	return ret