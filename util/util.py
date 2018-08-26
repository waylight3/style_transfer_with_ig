import time, os

def pgbar(data, pre='', post='', bar_icon='=', space_icon=' ', show_running_time=True, end='\r'):
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
		if show_running_time:
			print('%s [%s%s] [%3d%%] [%02dh%02dm%02ds] %s' % (pre, bar, space, hour, minute, second, 100 * (i + 1) // size, post), end=end)
		else:
			print('%s [%s%s] [%3d%%] %s' % (pre, bar, space, hour, minute, second, 100 * (i + 1) // size, post), end=end)
		yield d
