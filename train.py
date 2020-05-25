import os
import numpy as np
from keras.utils import np_utils
from models.ml import SVM_Model, MLP_Model
from models.dnn import LSTM_Model
import extract_feats.opensmile as of
import extract_feats.librosa as lf
from config import config
import misc.opts as opts

'''
train(): 训练模型

输入:
	model_name: 模型名称（svm / mlp / lstm）
	save_model_name: 保存模型的文件名
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出: 
    model: 训练好的模型
'''
def train(model_name: str, save_model_name: str, feature_method: str = 'o'):
    
    # 加载被 preprocess.py 预处理好的特征
    if(feature_method == 'o'):
        x_train, x_test, y_train, y_test = of.load_feature(feature_path = config.TRAIN_FEATURE_PATH_OPENSMILE, train = True)

    elif(feature_method == 'l'):
        x_train, x_test, y_train, y_test = lf.load_feature(feature_path = config.TRAIN_FEATURE_PATH_LIBROSA, train = True)

    # 创建模型
    if(model_name == 'svm'):
        model = SVM_Model()
    elif(model_name == 'mlp'):
        model = MLP_Model()
    elif(model_name == 'lstm'):
        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_test)

        model = LSTM_Model(input_shape = x_train.shape[1], num_classes = len(config.CLASS_LABELS))

        # 二维数组转三维（samples, time_steps, input_dim）
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # 训练模型
    print('---------------------------- Start Training ----------------------------')
    if(model_name == 'svm' or model_name == 'mlp'):
        model.train(x_train, y_train)
    elif(model_name == 'lstm'):
        model.train(x_train, y_train, x_test, y_val, n_epochs = config.epochs)
    print('------------------------------ End Training ------------------------------')

    # 验证模型
    model.evaluate(x_test, y_test)
    # 保存训练好的模型
    model.save_model(save_model_name)


if __name__ == '__main__':

    opt = opts.parse_train()
    train(model_name = opt.model_type, save_model_name = opt.model_name, feature_method = opt.feature)
    
    # train(model_name = "lstm", save_model_name = "LSTM_OPENSMILE", feature_method = 'l')