# CNN & LSTM
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from Common_Model import Common_Model
from Utilities import plotLine

# class CNN 和 class LSTM 继承了此类（实现了make_model方法）
class DNN_Model(Common_Model):
    '''
    __init__(): 初始化神经网络

    输入:
        input_shape(tuple): 张量形状
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
        best_acc = 0
        acc = []
        loss = []
        val_acc = []
        val_loss = []

        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        for i in range(n_epochs):
            # 每个epoch都随机排列训练数据
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

            history = self.model.fit(x_train, y_train, batch_size = 32, epochs = 1)
            # 训练集上的损失率和准确率
            acc.append(history.history['acc'])
            loss.append(history.history['loss'])
            # 验证集上的损失率和准确率
            val_loss_single, val_acc_single = self.model.evaluate(x_val, y_val)
            val_acc.append(val_acc_single)
            val_loss.append(val_loss_single)
            if val_acc_single > best_acc:
                best_acc = val_acc_single
        
        plotLine(acc, val_acc, 'Accuracy', 'acc')
        plotLine(loss, val_loss, 'Loss', 'loss')

        self.trained = True


    '''
    recognize_one(): 识别某个音频的情感

    输入:
        sample: 要预测的样本
    
    输出:
        预测结果，置信概率(int, numpy.ndarray)
    '''
    def recognize_one(self, sample):
        # 没有训练和加载过模型
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return np.argmax(self.model.predict(np.array([sample]))), self.model.predict(np.array([sample]))[0]

    def make_model(self):
        raise NotImplementedError()


class CNN_Model(DNN_Model):

    def __init__(self, **params):
        params['name'] = 'CNN'
        super(CNN_Model, self).__init__(**params)

    def make_model(self):
        self.model.add(Conv2D(8, (13, 13), input_shape=(self.input_shape[0], self.input_shape[1], 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))


class LSTM_Model(DNN_Model):

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM_Model, self).__init__(**params)

    def make_model(self):
        self.model.add(KERAS_LSTM(128, input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        # self.model.add(Dense(16, activation='tanh'))
        
