from keras.preprocessing import sequence
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from data_reader import *
import os


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
        self.model.add(LSTM(512, return_sequences=True))
        self.model.add(LSTM(512))
        self.model.add(Dense(self.n_words + 2, activation='softmax'))
        self.model.summary()
        self.model.compile('adam', 'categorical_crossentropy', metrics=['acc'])
        if self.load_model \
                and self.weight \
                and os.path.isfile(self.weight):
            self.model.load_weights(self.weight)
            print('Load model parameter successfully.')

    def train(self, X, Y, **kwargs):
        print('Start train:')
        self.model.fit(X, Y, callbacks=[
            ModelCheckpoint(self.weight),
            TensorBoard(),
        ], **kwargs)

    def generate(self, words, starts=None):
        header = []
        if starts:
            items = [text_2_words(starts)]
            for item in items:
                if item not in words:
                    raise Exception('Words not contains % s' % item)
                header.append(words.index(item))
        current = starts or ''
        result = header[:]
        input = sequence.pad_sequences([header], maxlen=maxlen, value=len(words))
        while True:
            outputs = self.model.predict(input)[0]
            output = choose_result_by_prob(outputs)
            if output == n_words:
                # 占位符重新生成
                continue
            if output == n_words + 1:
                # 结束符完毕
                break
            result.append(output)
            input = np.array([input[0][1:].tolist() + [output]])
            current += ' ' + to_word(output, words)
            sys.stdout.write('\r' + current)
        print()
        return result


train = True
n_generate = 1
maxlen = 100
n_words = 2000
n_embedding = 50
weight = 'data/model.h5'
load_model = True
step = 1
max_file = 10
batch_size = 32
epochs = 100
PATH = '/Users/puke/Documents/workSpace/PyCharm/Coding'

if __name__ == '__main__':
    X, Y, words = load_data(PATH, n_words, maxlen, step, max_file)
    model = LSTMModel(maxlen, len(words), n_embedding, weight, load_model)
    if train:
        model.train(X, Y, batch_size=batch_size, epochs=epochs)
    else:
        print('\nGenerated:')
        for _ in range(n_generate):
            model.generate(words, '')
