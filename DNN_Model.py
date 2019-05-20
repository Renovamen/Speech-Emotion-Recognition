# LSTM
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout
from Common_Model import Common_Model

# class LSTM 继承了此类
class DNN_Model(Common_Model):
    '''
    __init__(): 初始化神经网络

    输入:
        input_shape: 特征维度
        num_classes(int): 标签种类数量
    '''
    def __init__(self, input_shape, num_classes, **params):
        super(DNN_Model, self).__init__(**params)
        self.input_shape = input_shape
        self.model = Sequential()
        self.make_model()
        self.model.add(Dense(num_classes, activation = 'softmax'))
        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        print(self.model.summary(), file = sys.stderr)

    '''
    save_model(): 将模型权重以 model_name.h5 和 model_name.json 命名存储在 /Models 目录下
    '''
    def save_model(self, model_name):
        h5_save_path = 'Models/' + model_name + '.h5'
        self.model.save_weights(h5_save_path)

        save_json_path = 'Models/' + model_name + '.json'
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train (numpy.ndarray): 训练集样本
        y_train (numpy.ndarray): 训练集标签
        x_val (numpy.ndarray): 测试集样本
        y_val (numpy.ndarray): 测试集标签
        n_epochs (int): epoch数

    '''
    def train(self, x_train, y_train, x_val = None, y_val = None, n_epochs = 50):
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        for i in range(n_epochs):
            # 每个epoch都随机排列训练数据
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
            self.model.fit(x_train, y_train, batch_size = 32, epochs = 1)
            # 计算损失率和准确率
            loss, acc = self.model.evaluate(x_val, y_val)
        self.trained = True


    '''
    predict(): 识别音频的情感

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 识别结果
    '''
    def predict(self, sample):
        # 没有训练和加载过模型
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)

        return np.argmax(self.model.predict(sample), axis=1)


    def make_model(self):
        raise NotImplementedError()


class LSTM_Model(DNN_Model):

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM_Model, self).__init__(**params)

    def make_model(self):
        self.model.add(KERAS_LSTM(128, input_shape=(1, self.input_shape)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))
        
