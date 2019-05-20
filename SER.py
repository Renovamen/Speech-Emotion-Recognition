import numpy as np
from keras.utils import np_utils
import os

from ML_Model import SVM_Model, MLP_Model
from DNN_Model import LSTM_Model

from Utils import load_model, Radar

import Opensmile_Feature as of
import Librosa_Feature as lf

DATA_PATH = 'DataSet/CASIA'
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
CONFIG = 'IS10_paraling'
OPENSMILE_PATH = '/Users/zou/opensmile-2.3.0'
TRAIN_FEATURE_PATH = 'Feature/feature.csv'
PREDICT_FEATURE_PATH = 'Feature/test.csv'
NEW_TRAIN_FEATURE_PATH = 'Feature/feature.p'
NEW_PREDICT_FEATURE_PATH = 'Feature/test.p'

'''
Train(): 训练模型

输入:
	model_name: 模型名称（SVM / MLP / LSTM）
	save_model_name: 保存模型的文件名
	epochs: epoch 数量（SVM 和 MLP 模型不需要传该参数）
输出：
	model: 训练好的模型
'''
def Train(model_name: str, save_model_name: str, epochs: int = 50):
    
    # 提取特征
    # x_train, x_test, y_train, y_test = of.get_data(OPENSMILE_PATH, DATA_PATH, TRAIN_FEATURE_PATH, CONFIG, CLASS_LABELS, train = True)
    # x_train, x_test, y_train, y_test = of.load_feature(feature_path = TRAIN_FEATURE_PATH, train = True)
    
    # x_train, x_test, y_train, y_test = lf.get_data(DATA_PATH, NEW_TRAIN_FEATURE_PATH, CLASS_LABELS, train = True)
    x_train, x_test, y_train, y_test = lf.load_feature(feature_path = NEW_TRAIN_FEATURE_PATH, train = True)

    # 创建模型
    if(model_name == 'SVM'):
        model = SVM_Model()
    elif(model_name == 'MLP'):
        model = MLP_Model()
    elif(model_name == 'LSTM'):
        y_train = np_utils.to_categorical(y_train)
        y_val = np_utils.to_categorical(y_test)

        model = LSTM_Model(input_shape = x_train.shape[1], num_classes = len(CLASS_LABELS))

        # 二维数组转三维（samples, time_steps, input_dim）
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

    # 训练模型
    print('-------------------------------- Start --------------------------------')
    if(model_name == 'SVM' or model_name == 'MLP'):
        model.train(x_train, y_train)
    elif(model_name == 'LSTM'):
        model.train(x_train, y_train, x_test, y_val, n_epochs = epochs)

    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)
    print('---------------------------------- End ----------------------------------')

    return model


'''
Predict(): 预测音频情感
输入:
	model: 已加载或训练的模型
	model_name: 模型名称
	save_model_name: 保存模型的文件名
输出：
	file_path: 要预测的文件路径
'''
def Predict(model, model_name: str, file_path: str):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    # test_feature = of.load_feature(feature_path = PREDICT_FEATURE_PATH, train = False)
    # test_feature = of.get_data(OPENSMILE_PATH, file_path, PREDICT_FEATURE_PATH, CONFIG, CLASS_LABELS, train = False)
    
    # test_feature = lf.load_feature(feature_path = NEW_PREDICT_FEATURE_PATH, train = False)
    test_feature = lf.get_data(file_path, NEW_PREDICT_FEATURE_PATH, CLASS_LABELS, train = False)
    
    if(model_name == 'LSTM'):
        # 二维数组转三维（samples, time_steps, input_dim）
        test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))
    
    result = model.predict(test_feature)
    if(model_name == 'LSTM'):
        result = np.argmax(result)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS)



model = Train(model_name = "LSTM", save_model_name = "LSTM_LIBROSA", epochs = 10)
# 加载模型
# model = load_model(load_model_name = "LSTM_LIBROSA", model_name = "LSTM")
Predict(model, model_name = "LSTM", file_path = "Test/201-happy-liuchanhg.wav")