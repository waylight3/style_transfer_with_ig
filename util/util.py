import os

def pgbar(data, pre='', post='', bar_icon='=', space_icon=' ', end='\r'):
	'PRE [PGBAR] [000%] POST'

	size = len(data)
	tsize, _ = os.get_terminal_size()
	bsize = tsize - len(pre) - len(post) - 12

	for i, d in enumerate(data):
		bar = bar_icon * (bsize * (i + 1) // size // len(bar_icon))
		space = space_icon * (bsize - len(bar))
		print('%s [%s%s] [%3d%%] %s' % (pre, bar, space, 100 * (i + 1) // size, post), end=end)
		yield d
