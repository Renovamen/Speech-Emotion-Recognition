import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense
from ..common import Common_Model
from utils.common import plotCurve

# class LSTM, CNN1D 继承了此类
class DNN_Model(Common_Model):
    '''
    __init__(): 初始化神经网络

    输入:
        input_shape: 特征维度
        num_classes(int): 标签种类数量
        lr(float): 学习率
    '''
    def __init__(self, input_shape, num_classes, lr, **params):
        super(DNN_Model, self).__init__()

        self.input_shape = input_shape
        
        self.model = Sequential()
        self.make_model(**params)
        self.model.add(Dense(num_classes, activation = 'softmax'))

        optimzer = keras.optimizers.Adam(lr = lr)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimzer, metrics = ['accuracy'])
       
        print(self.model.summary(), file = sys.stderr)


    '''
    save_model(): 将模型存储在 config.checkpoint_path 路径下

    输入:
        config(Class)
    '''
    def save_model(self, config):
        h5_save_path = config.checkpoint_path + config.checkpoint_name + '.h5'
        self.model.save_weights(h5_save_path)

        save_json_path = config.checkpoint_path + config.checkpoint_name + '.json'
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())


    def reshape_input(self):
        NotImplementedError()


    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train(numpy.ndarray): 训练集样本
        y_train(numpy.ndarray): 训练集标签
        x_val(numpy.ndarray): 测试集样本
        y_val(numpy.ndarray): 测试集标签
        batch_size(int): 批大小
        n_epochs(int): epoch 数
    '''
    def train(self, x_train, y_train, x_val = None, y_val = None, 
                batch_size = 32, n_epochs = 50):
 
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        
        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        history = self.model.fit(
            x_train, y_train, 
            batch_size = batch_size, 
            epochs = n_epochs,
            shuffle = True, # 每个 epoch 开始前随机排列训练数据
            validation_data = (x_val, y_val)
        )

        # 训练集上的损失和准确率
        acc = history.history['acc']
        loss = history.history['loss']
        # 验证集上的损失和准确率
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']

        plotCurve(acc, val_acc, 'Accuracy', 'acc')
        plotCurve(loss, val_loss, 'Loss', 'loss')
        
        self.trained = True


    '''
    predict(): 识别音频的情感

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 识别结果
    '''
    def predict(self, sample):
        sample = self.reshape_input(sample)
        
        # 没有训练和加载过模型
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)

        return np.argmax(self.model.predict(sample), axis=1)


    def make_model(self):
        raise NotImplementedError()