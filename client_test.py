import random
import time
from text.util import *

if __name__ == '__main__':
    size = 100000
    word2index = {str(i): i for i in range(size)}
    words = [str(i) for i in range(size)]

    start = time.time()
    for i in range(5000000):
        # a = word2index[str(i % size)]
        a = words.index(str(i % size))
        # show_progress(i + 1, 5000000)
    print(time.time() - start)
