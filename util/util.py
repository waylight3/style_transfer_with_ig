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
		if i % (size // total_display) == 0 or i == size - 1:
			print('%s [%s%s] [%3d%%]%s %s' % (pre, bar, space, 100 * (i + 1) // size, running_time, post), end=end)
		if i == size - 1:
			print('')
		yield d
