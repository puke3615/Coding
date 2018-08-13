from text_preprocessor.util import *
from collections import Counter
import numpy as np
import os


def get_all_files(path, file_selector=lambda filepath: True, dir_selector=lambda dirpath: True, result=None):
    """
    遍历文件
    :param path: 遍历目录
    :param file_selector: 文件选择器
    :param dir_selector:  文件夹选择器
    :param result: 结果集
    :return: 文件列表
    """
    result = [] if result is None else result
    if not os.path.exists(path):
        raise Exception('File "%s" not found' % path)
    isfile = os.path.isfile(path)
    if isfile:
        if not file_selector or file_selector(path):
            result.append(path)
    else:
        if not dir_selector or dir_selector(path):
            for sub in os.listdir(path):
                sub_path = os.path.join(path, sub)
                get_all_files(sub_path, file_selector, dir_selector, result)
    return result


def parse_text(filepath, sep):
    """
    解析文本中的word, 如
    >>>[['a', 'b', 'c']]
    :param filepath: 文件路径
    :param sep: 分割符
    :return: word列表的列表
    """
    assert os.path.isfile(filepath)
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
        if sep:
            return text.split(sep)
        else:
            return [[c for c in text]]


def words_cmp(a, b):
    """
    word比较器
    :param a: (word, 词频)
    :param b: (word, 词频)
    :return: [-1|0|1]
    """
    a_word, a_freq = a
    b_word, b_freq = b
    if a_freq != b_freq:
        return b_freq - a_freq
    elif a_word != b_word:
        return -1 if a_word < b_word else 1
    else:
        return 0


class Dataset:
    def __init__(self, path, maxlen, batch_size=32, seed=None, step=1, sep=None, min_freq=None, max_words=None,
                 max_data=None,
                 postfix=None, max_file=None, ignore_file=None, ignore_dir=None, ignore_file_error=True,
                 file_selector=None, dir_selector=None, word_parser=None):
        self.path = path
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.step = step
        self.sep = sep
        self.min_freq = min_freq
        self.max_words = max_words
        self.max_data = max_data
        self.postfix = postfix
        self.max_file = max_file
        self.ignore_file = ignore_file
        self.ignore_dir = ignore_dir
        self.ignore_file_error = ignore_file_error
        self.file_selector = file_selector
        self.dir_selector = dir_selector
        self.word_parser = word_parser

        self.index_padding = None
        self.index_end = None
        self.words = None
        self.word2index = None

    def load_data(self, show_mapping_progress=False):

        def default_file_selector(filepath):
            filename = os.path.basename(filepath)
            if self.ignore_file and filename in self.ignore_file:
                return False
            return not self.postfix or any([filename.endswith(p) for p in self.postfix])

        def default_dir_selector(dirpath):
            return not self.ignore_dir or os.path.basename(dirpath) not in self.ignore_dir

        file_selector = self.file_selector or default_file_selector
        dir_selector = self.dir_selector or default_dir_selector
        word_parser = self.word_parser or (lambda filepath: parse_text(filepath, self.sep))

        # read all files
        files = get_all_files(self.path, file_selector, dir_selector)
        n_files = len(files)
        if self.max_file:
            files = files[:self.max_file]
        print('Total %d files, select %d files.' % (n_files, len(files)))
        # collect words
        origin_data = []
        counter = Counter()
        n_files = len(files)
        for i, file in enumerate(files):
            show_progress(i + 1, n_files, 'Collect words', file)
            try:
                file_words = word_parser(file)
            except Exception as e:
                if self.ignore_file_error:
                    continue
                else:
                    raise e
            if not file_words:
                continue
            origin_data.extend(file_words)
            for item_words in file_words:
                for word in item_words:
                    counter[word] += 1
        print()

        # mapping words to indexes
        n_all_words = len(counter)
        if self.max_words:
            counter = counter.most_common(self.max_words)
        counter = dict(counter)
        if self.min_freq:
            counter = dict({k: v for k, v in counter.items() if v >= self.min_freq})
        self.n_words = len(counter)
        print('Found %d words, select %d words.' % (n_all_words, self.n_words))

        import functools
        # self.words = list(sorted(counter, key=functools.cmp_to_key(words_cmp)))
        # self.words = list(map(lambda x: x[0], sorted(counter, key=functools.cmp_to_key(words_cmp))))
        self.words = list(sorted(counter, key=lambda w: '%09d_%s' % (counter[w], w), reverse=True))
        self.word2index = {w: i for i, w in enumerate(self.words)}

        self.index_padding = self.n_words
        self.index_end = self.n_words + 1

        xy = []
        index = 0
        n_origin_data = sum(map(len, origin_data)) + len(origin_data)
        n_data = min(self.max_data, n_origin_data) if self.max_data else n_origin_data
        print('Total %d data, select %d data.' % (n_origin_data, n_data))

        get_index = lambda word: self.word2index[word] if word in self.words else self.index_padding
        for items in origin_data:
            n_items = len(items)
            for i in range(0, n_items + 1, self.step):
                if index >= n_data:
                    break
                input_words = items[max(0, i - self.maxlen): i]
                input = list(map(get_index, input_words))
                output = self.index_end if i == len(items) else get_index(items[i])
                if show_mapping_progress:
                    show_progress(index + 1, n_origin_data, 'Mapping data',
                                  )
                    # input_words)
                    # '%s ==> %s' % (str(input_words), str(input)))
                xy.append((input, output))
                index += 1
        show_mapping_progress and print()

        # shuffle initial data
        if self.seed:
            np.random.seed(self.seed)
        np.random.shuffle(xy)

        if n_data < self.batch_size:
            raise Exception('Too few data.')

        from keras.preprocessing import sequence
        from keras.utils import np_utils
        steps_per_epoch = n_data // self.batch_size
        yield steps_per_epoch
        epoch = 1
        while True:
            # print('\n>>>Load data on epoch %d <<<' % epoch)
            flags = list(range(n_data))
            for step in range(steps_per_epoch):
                choose = np.random.choice(flags, self.batch_size, replace=False)
                xs, ys = [], []
                for index in choose:
                    flags.remove(index)
                    x, y = xy[index]
                    xs.append(x)
                    ys.append(y)
                X = sequence.pad_sequences(xs, maxlen=self.maxlen, value=self.index_padding)
                Y = np_utils.to_categorical(ys, self.n_words + 2)
                yield X, Y
            epoch += 1

    def to_words(self, indexes, join_str=None, v_padding='å'):
        join_str = join_str or ''
        mapping = lambda i: self.to_word(i, v_padding)
        return join_str.join(map(mapping, indexes))

    def to_word(self, index, v_padding='å'):
        return self.words[index] if index < len(self.words) else v_padding


if __name__ == '__main__':
    path = '../../LSTMDemo/data'
    dataset = Dataset(
        path, maxlen=10,
        max_data=100,
        postfix=['.txt'],
        max_file=1,
    )
    generator = dataset.load_data(False)
    steps_per_epoch = next(generator)
    print(dataset.words)

    for x, y in generator:
        print('\n'.join(
            [dataset.to_words(x_item) + ' ==> ' + dataset.to_word(np.argmax(y_item))
             for x_item, y_item in zip(x, y)]))
        print('\n\n\n')
