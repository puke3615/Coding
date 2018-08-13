from keras.preprocessing import sequence
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from text import *
from data_reader import *
import os


def choose_result_by_best(predict):
    return np.argmax(predict)


def choose_result_by_prob(predict):
    # 取词逻辑
    # 将predict累加求和
    t = np.cumsum(predict)
    # 求出预测可能性的总和
    s = np.sum(predict)
    # 返回将0~s的随机值插值到t中的索引值
    # 由于predict各维度对应的词向量是按照训练数据集的频率进行排序的
    # 故P(x|predict[i]均等时) > P(x + δ), 即达到了权衡优先取前者和高概率词向量的目的
    return int(np.searchsorted(t, np.random.rand(1) * s))


class LSTMModel:
    def __init__(self, maxlen, n_words, n_embedding, weight, load_model=True):
        self.maxlen = maxlen
        self.n_words = n_words
        self.n_embedding = n_embedding
        self.weight = weight
        dir = os.path.dirname(os.path.abspath(weight))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        self.load_model = load_model
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.n_words + 2, self.n_embedding, input_length=self.maxlen))
        self.model.add(BatchNormalization())
        # self.model.add(LSTM(512, return_sequences=True))
        # self.model.add(BatchNormalization())
        self.model.add(LSTM(512, return_sequences=False))
        self.model.add(BatchNormalization())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(self.n_words + 2, activation='softmax'))
        self.model.summary()
        self.model.compile('rmsprop', 'categorical_crossentropy', metrics=['acc'])
        if self.load_model \
                and self.weight \
                and os.path.isfile(self.weight):
            self.model.load_weights(self.weight)
            print('Load model parameter successfully.')
        else:
            print('No model parameter found.')

    def train(self, X, Y, **kwargs):
        print('Start train:')
        self.model.fit(X, Y, callbacks=[
            ModelCheckpoint(self.weight),
            TensorBoard(),
        ], **kwargs)

    def train_generator(self, generator, steps_one_epoch, epochs, **kwargs):
        print('Start train:')
        self.model.fit_generator(generator, steps_one_epoch, epochs, callbacks=[
            ModelCheckpoint(self.weight),
            TensorBoard(),
        ], **kwargs)

    def generate(self, words, maxlen, starts=None, choose_best=False, v_join=''):
        header = []
        if starts:
            items = [text_2_words(starts)[0]]
            for item in items:
                if item not in words:
                    raise Exception('Words not contains % s' % item)
                header.append(words.index(item))
        current = starts or ''
        result = header[:]
        n_words = len(words)
        input = sequence.pad_sequences([header], maxlen=maxlen, value=n_words)
        while True:
            outputs = self.model.predict(input)[0]
            choose = lambda: choose_result_by_best(outputs) if choose_best else choose_result_by_prob(outputs)
            output = choose()
            choose_times = 0
            max_choose_times = 10
            while output == n_words:
                if choose_times >= max_choose_times - 1:
                    break
                # 占位符重新选择
                output = choose()
                choose_times += 1
            if choose_times == max_choose_times:
                continue
            if output == n_words + 1:
                # 结束符完毕
                break
            result.append(output)
            input = np.array([input[0][1:].tolist() + [output]])
            current += v_join + to_word(output, words)
            sys.stdout.write('\r' + current)
        print()
        return result


train = False
choose_best = False
n_generate = 100
maxlen = 32
n_words = 2000
min_freq = 20
n_embedding = 50
# weight = 'data/100-0.024574.hdf5'
# weight = 'data/{epoch:02d}-{loss:4f}.hdf5'
weight = 'data/model.hdf5'
load_model = True
step = 1
max_file = None
max_data = None
batch_size = 32
epochs = 100
v_join = ''
# PATH = './'
PATH = './data'
# PATH = '../LSTMDemo/data'
postfix = '.txt'

if __name__ == '__main__':
    dataset = Dataset(
        PATH,
        maxlen=maxlen,
        sep='\n\n',
        step=step,
        max_words=n_words,
        postfix=[postfix],
        batch_size=batch_size,
        max_data=max_data,
        min_freq=min_freq,
    )
    generator = dataset.load_data(True)
    steps_one_epoch = next(generator)
    words = dataset.words
    print(words)
    # exit(0)

    # X, Y, words = load_data(PATH, n_words, maxlen, step, max_file, postfix, max_data)
    model = LSTMModel(maxlen, len(words), n_embedding, weight, load_model)
    if train:
        # model.train(X, Y, batch_size=batch_size, epochs=epochs)
        model.train_generator(
            generator,
            steps_one_epoch=steps_one_epoch // 100,
            epochs=epochs,
        )
    else:
        print('\nGenerated:')
        for _ in range(n_generate):
            model.generate(words, maxlen, '', choose_best, v_join)
