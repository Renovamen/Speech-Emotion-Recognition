import numpy as np
from keras.utils import np_utils
import os

from ML_Model import SVM_Model, MLP_Model
from DNN_Model import LSTM_Model

from Utils import load_model, Radar, playAudio

import Opensmile_Feature as of
import Librosa_Feature as lf
from Config import Config

'''
Train(): 训练模型

输入:
	model_name: 模型名称（SVM / MLP / LSTM）
	save_model_name: 保存模型的文件名
    if_load: 是否加载已有特征（True / False）
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出：
	model: 训练好的模型
'''
def Train(model_name: str, save_model_name: str, if_load: bool = True, feature_method: str = 'opensmile'):
    
    # 提取特征
    if(feature_method == 'o'):
        if(if_load == True):
            x_train, x_test, y_train, y_test = of.load_feature(feature_path = Config.TRAIN_FEATURE_PATH_OPENSMILE, train = True)
        else:
            x_train, x_test, y_train, y_test = of.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_OPENSMILE, train = True)
    
    elif(feature_method == 'l'):
        if(if_load == True):
            x_train, x_test, y_train, y_test = lf.load_feature(feature_path = Config.TRAIN_FEATURE_PATH_LIBROSA, train = True)
        else:
            x_train, x_test, y_train, y_test = lf.get_data(Config.DATA_PATH, Config.TRAIN_FEATURE_PATH_LIBROSA, train = True)

    # 创建模型
    if(model_name == 'svm'):
        model = SVM_Model()
    elif(model_name == 'mlp'):
        model = MLP_Model()
    elif(model_name == 'lstm'):
        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_test)

        model = LSTM_Model(input_shape = x_train.shape[1], num_classes = len(Config.CLASS_LABELS))

        # 二维数组转三维（samples, time_steps, input_dim）
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # 训练模型
    print('-------------------------------- Start --------------------------------')
    if(model_name == 'svm' or model_name == 'mlp'):
        model.train(x_train, y_train)
    elif(model_name == 'lstm'):
        model.train(x_train, y_train, x_test, y_val, n_epochs = Config.epochs)

    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)
    print('---------------------------------- End ----------------------------------')

    return model


'''
Predict(): 预测音频情感
输入:
	model: 已加载或训练的模型
	model_name: 模型名称
	file_path: 要预测的文件路径
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）
输出：
    预测结果和置信概率
'''
def Predict(model, model_name: str, file_path: str, feature_method: str = 'Opensmile'):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    playAudio(file_path)

    if(feature_method == 'o'):
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(file_path, Config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
        test_feature = of.load_feature(Config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
    elif(feature_method == 'l'):
        test_feature = lf.get_data(file_path, Config.PREDICT_FEATURE_PATH_LIBROSA, train = False)
    
    if(model_name == 'lstm'):
        # 二维数组转三维（samples, time_steps, input_dim）
        test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))
    
    result = model.predict(test_feature)
    if(model_name == 'lstm'):
        result = np.argmax(result)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', Config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob)



# model = Train(model_name = "lstm", save_model_name = "LSTM_OPENSMILE_1", if_load = True, feature_method = 'o')
# 加载模型
# model = load_model(load_model_name = "LSTM_OPENSMILE", model_name = "lstm")
# Predict(model, model_name = "lstm", file_path = "Test/neutral.wav", feature_method = 'o')