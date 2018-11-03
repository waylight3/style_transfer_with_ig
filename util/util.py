from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

def make_batch(data_x, data_x_len, data_y, data_y_len, data_mask, data_star, batch_size):
	data_size = len(data_x)
	ret = []
	for batch in range(data_size // batch_size):
		feed_dict = {'X':data_x[batch*batch_size:(batch+1)*batch_size], 'X_len':data_x_len[batch*batch_size:(batch+1)*batch_size], 'Y':data_y[batch*batch_size:(batch+1)*batch_size], 'Y_len':data_y_len[batch*batch_size:(batch+1)*batch_size], 'Y_mask':data_mask[batch*batch_size:(batch+1)*batch_size], 'Star':data_star[batch*batch_size:(batch+1)*batch_size]}
		max_len = max(feed_dict['X_len'])
		for i in range(batch_size):
			feed_dict['X'][i] = feed_dict['X'][i][:max_len]
			feed_dict['Y'][i] = feed_dict['Y'][i][:max_len]
			feed_dict['Y_mask'][i] = feed_dict['Y_mask'][i][:max_len]
			feed_dict['X_len'][i] = min(feed_dict['X_len'][i], max_len)
			feed_dict['Y_len'][i] = min(feed_dict['Y_len'][i], max_len)
		ret.append([batch, feed_dict])
	return ret

def seq2seq_accuracy(logits, targets, weights):
	batch_size = len(logits)
	total_acc = 0.0
	for bc in range(batch_size):
		acc = 0.0
		cnt = 0.0
		for i in range(len(logits[bc])):
			if weights[bc][i] < 0.01: continue
			cnt += 1.0
			if targets[bc][i] != logits[bc][i]: continue
			acc += 1.0
		acc /= cnt
		total_acc += acc
	total_acc /= batch_size
	return total_acc

def seq2seq_bleu(logits, targets, end):
	batch_size = len(logits)
	total_bleu = 0.0
	for bc in range(batch_size):
		logits_end = len(logits[bc])
		for i in range(len(logits[bc])):
			if logits[bc][i] == end:
				logits_end = i
				break
		targets_end = len(targets[bc])
		for i in range(len(targets[bc])):
			if targets[bc][i] == end:
				targets_end = i
				break
		if logits_end <= 1 or targets_end == 0: continue
		score = sentence_bleu([targets[bc][:targets_end]], logits[bc][:logits_end], weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method4)
		total_bleu += score
	total_bleu /= batch_size
	return total_bleu

def interpolate(v1, v2, step):
	ret = []
	for i in range(1, step + 1):
		ret.append([v1[j] + (v2[j] - v1[j]) * i / step for j in range(len(v1))])
	return ret

# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
			cbar_kw={}, cbarlabel="", **kwargs):
	"""
	Create a heatmap from a numpy array and two lists of labels.

	Arguments:
		data       : A 2D numpy array of shape (N,M)
		row_labels : A list or array of length N with the labels
					 for the rows
		col_labels : A list or array of length M with the labels
					 for the columns
	Optional arguments:
		ax         : A matplotlib.axes.Axes instance to which the heatmap
					 is plotted. If not provided, use current axes or
					 create a new one.
		cbar_kw    : A dictionary with arguments to
					 :meth:`matplotlib.Figure.colorbar`.
		cbarlabel  : The label for the colorbar
	All other arguments are directly passed on to the imshow call.
	"""

	if not ax:
		ax = plt.gca()

	# Plot the heatmap
	im = ax.imshow(data, **kwargs)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

	# We want to show all ticks...
	ax.set_xticks(np.arange(data.shape[1]))
	ax.set_yticks(np.arange(data.shape[0]))
	# ... and label them with the respective list entries.
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)

	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False,
				   labeltop=True, labelbottom=False)

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
			 rotation_mode="anchor")

	# Turn spines off and create white grid.
	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)

	return im, cbar

# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def formatted_heatmap(im, data=None, valfmt="{x:.2f}",
					 textcolors=["black", "white"],
					 threshold=None, **textkw):
	"""
	A function to annotate a heatmap.

	Arguments:
		im         : The AxesImage to be labeled.
	Optional arguments:
		data       : Data used to annotate. If None, the image's data is used.
		valfmt     : The format of the annotations inside the heatmap.
					 This should either use the string format method, e.g.
					 "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
		textcolors : A list or array of two color specifications. The first is
					 used for values below a threshold, the second for those
					 above.
		threshold  : Value in data units according to which the colors from
					 textcolors are applied. If None (the default) uses the
					 middle of the colormap as separation.

	Further arguments are passed on to the created text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)

	return texts

# https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
def annotate_heatmap(im, data=None, texts=None,
					 textcolors=["white", "black"],
					 threshold=None, threshold2=None, **textkw):
	"""
	A function to annotate a heatmap.

	Arguments:
		im         : The AxesImage to be labeled.
	Optional arguments:
		data       : Data used to annotate. If None, the image's data is used.
		texts      : The texts of the annotations inside the heatmap.
		textcolors : A list or array of two color specifications. The first is
					 used for values below a threshold, the second for those
					 above.
		threshold  : Value in data units according to which the colors from
					 textcolors are applied. If None (the default) uses the
					 middle of the colormap as separation.

	Further arguments are passed on to the created text labels.
	"""

	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()

	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max())/2.
	if threshold2 is not None:
		threshold2 = im.norm(threshold2)

	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)

	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(threshold < im.norm(data[i, j]) < threshold2)])
			text = im.axes.text(j, i, texts[i, j], **kw)

	return texts

def rescale(vec, full=1, base='zero'):
	min_val = min(vec)
	max_val = max(vec)
	if base == 'min':
		base = min_val
	elif base == 'max':
		base = max_val
	elif base == 'zero':
		base = 0
	return [full * (v - base) / (max_val - min_val) for v in vec]

# http://chris35wills.github.io/matplotlib_diverging_colorbar/
# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
