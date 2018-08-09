from collections import Counter
import sys
import os

# PATH = '/Users/puke/Documents/workSpace/PyCharm/Coding'
PATH = '/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5'


def get_all_files(path, file_selector=lambda filepath: True, dir_selector=lambda dirpath: True, result=None):
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


def load_data(path=PATH, n_words=2000, maxlen=100, step=1, max_file=None, postfix='.py'):
    files = get_all_files(path, file_selector=lambda filepath: filepath.endswith(postfix),
                          dir_selector=lambda dirpath: os.path.basename(dirpath) not in ['.git'])
    n_files = len(files)
    if max_file:
        files = files[:max_file]
    print('Found %d files, select %d files.' % (n_files, len(files)))
    all_file_words = []
    counter = Counter()
    for i, file in enumerate(files):
        current = i + 1
        total = len(files)
        rate = current * 100. / total
        sys.stdout.write('\r[%02d%%] Collect words: %d/%d' % (rate, current, total))
        with open(file, encoding='utf-8') as f:
            try:
                text = f.read().strip()
            except:
                continue
            if not text:
                continue
            file_words = []
            for word in text_2_words(text):
                # if word is '':
                #     continue
                file_words.append(word)
                counter[word] += 1
            all_file_words.append(file_words)
    print()
    n_all_words = len(counter)
    counter = dict(counter.most_common(n_words))
    n_words = len(counter)
    words = list(sorted(counter, key=lambda w: '%05d_%s' % (counter[w], w), reverse=True))
    print('Total %d words, select %d words.' % (n_all_words, len(counter)))

    # print(counter)
    # for word in sorted(counter, key=lambda x: -counter[x]):
    #     print('%s: %d' % (word, counter[word]))

    v_padding = n_words
    v_end = n_words + 1
    get_index = lambda word: words.index(word) if word in words else v_padding

    x = []
    y = []
    for f, file_words in enumerate(all_file_words):
        current = f + 1
        total = len(all_file_words)
        rate = current * 100. / total
        sys.stdout.write('\r[%02d%%] To index: %d/%d' % (rate, current, total))
        file_indexes = list(map(get_index, file_words))
        for i in range(0, len(file_words) + 1, step):
            input = file_indexes[max(i - maxlen, 0): i]
            output = v_end if i == len(file_indexes) else file_indexes[i]
            x.append(input)
            y.append(output)
    print()

    print('Preprocess...')
    from keras.preprocessing import sequence
    from keras.utils import np_utils
    X = sequence.pad_sequences(x, maxlen=maxlen, value=v_padding)
    Y = np_utils.to_categorical(y)

    return X, Y, words


def text_2_words(text):
    # return text.split(' ')
    return list(text)


def to_words(indexes, words, v_padding='å'):
    mapping = lambda i: to_word(i, words, v_padding)
    return ''.join(map(mapping, indexes))


def to_word(index, words, v_padding='å'):
    return words[index] if index < len(words) else v_padding


if __name__ == '__main__':
    X, Y, words = load_data('data/', max_file=10, maxlen=20)
    print('X.shape = %s' % str(X.shape))
    print('Y.shape = %s' % str(Y.shape))
    x_data = [to_words(x, words) for x in X]
    # print('\n'.join(x_data))
    print(words)
