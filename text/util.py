import sys


def get_process(current, total):
    percent = 100 * current / total
    return '%.1f%%' % percent


def show_progress(current, total, name=None, info=None):
    name = (name + ': ') if name else ''
    info = info or ''
    process = get_process(current, total)
    sys.stdout.write('\r%s[%s] %s' % (name, process, str(info)))


if __name__ == '__main__':
    import time

    size = 1000
    for i in range(size):
        time.sleep(.01)
        show_progress(i + 1, size, info='Process...')
