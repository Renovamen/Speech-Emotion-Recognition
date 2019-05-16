import numpy as np
import os

from ML_Model import SVM_Model
from ML_Model import MLP_Model

from Utilities import load_feature
from Utilities import get_data
from Utilities import load_model
from Utilities import Radar

DATA_PATH = 'DataSet/CASIA'
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
CONFIG = 'IS10_paraling'
OPENSMILE_PATH = '/Users/zou/opensmile-2.3.0'
TRAIN_FEATURE_PATH = 'Feature/feature.csv'
PREDICT_FEATURE_PATH = 'Feature/test.csv'

'''
---------------------------- 训练模型 ----------------------------
'''
def Train(model_name: str, save_model_name: str):
    
    # 创建模型
    if(model_name == 'SVM'):
        model = SVM_Model()
    elif(model_name == 'MLP'):
        model = MLP_Model()

    # 提取特征
    # x_train, x_test, y_train, y_test = get_data(OPENSMILE_PATH, DATA_PATH, TRAIN_FEATURE_PATH, CONFIG, CLASS_LABELS, train = True)
    x_train, x_test, y_train, y_test = load_feature(feature_path = TRAIN_FEATURE_PATH, train = True)

    # 训练模型
    print('-------------------------------- Start --------------------------------')
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model(save_model_name)
    print('---------------------------------- End ----------------------------------')

    return model


'''
---------------------------- 预测音频情感 ----------------------------
'''
def Predict(model, file_path: str):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    # test_feature = load_feature(feature_path = PREDICT_FEATURE_PATH, train = False)
    test_feature = get_data(OPENSMILE_PATH, file_path, PREDICT_FEATURE_PATH, CONFIG, CLASS_LABELS, train = False)
    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS)



model = Train(model_name = "SVM", save_model_name = "SVM1")
# ---------------------------- 加载模型 ----------------------------
# model = load_model(model_name = "MLP1", load_model = "ML")
Predict(model, "test/201-neutral-liuchanhg.wav")