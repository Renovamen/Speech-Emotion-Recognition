from keras.layers import LSTM as KERAS_LSTM
from keras.layers import Dense, Dropout
import numpy as np
from .dnn import DNN_Model

# LSTM
class LSTM(DNN_Model):
    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)
    
    def reshape_input(self, data):
        # 二维数组转三维 
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats) 
        # time_steps * input_size = n_feats
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data

    '''
    搭建 LSTM

    输入:
        rnn_size(int): LSTM 隐藏层大小
        hidden_size(int): 全连接层大小
        dropout(float)
    '''
    def make_model(self, rnn_size, hidden_size, dropout = 0.5, **params):
        self.model.add(KERAS_LSTM(rnn_size, input_shape=(1, self.input_shape))) # (time_steps = 1, n_feats)
        self.model.add(Dropout(dropout))
        self.model.add(Dense(hidden_size, activation='relu'))
        # self.model.add(Dense(rnn_size, activation='tanh'))