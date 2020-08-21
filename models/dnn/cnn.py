from keras.layers import Dense, Dropout, Flatten, Conv1D, \
    BatchNormalization, Activation, MaxPooling1D
import numpy as np
from .dnn import DNN_Model

# 1D CNN
class CNN1D(DNN_Model):
    def __init__(self, **params):
        params['name'] = 'CNN1D'
        super(CNN1D, self).__init__(**params)
    
    def reshape_input(self, data):
        # 二维数组转三维 
        # (n_samples, n_feats) -> (n_samples, n_feats, 1)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data

    '''
    搭建 1D CNN

    输入:
        n_kernels(int): 卷积核数量
        kernel_sizes(list): 每个卷积层的卷积核大小，列表长度为卷积层数量
        hidden_size(int): 全连接层大小
        dropout(float)
    '''
    def make_model(self, n_kernels, kernel_sizes, hidden_size, dropout = 0.5, **params):
        for size in kernel_sizes:
            self.model.add(Conv1D(
                filters = n_kernels,
                kernel_size = size, 
                padding = 'same', 
                input_shape = (self.input_shape, 1)
            ))  # 卷积层
            self.model.add(BatchNormalization(axis = -1))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(dropout))

        self.model.add(Flatten())
        self.model.add(Dense(hidden_size))  # 全连接层
        self.model.add(BatchNormalization(axis = -1))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dropout))