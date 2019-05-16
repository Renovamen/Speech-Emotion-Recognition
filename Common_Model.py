import sys
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score


class Common_Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False # 模型是否已训练

    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train: 训练集样本
        y_train: 训练集标签
        x_val: 测试集样本
        y_val: 测试集标签

    '''
    def train(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError()

    '''
    predict(): 识别音频的情感

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 识别结果（标签）的list
    '''
    def predict(self, samples):
        raise NotImplementedError()
        

    '''
    predict_proba(): 音频的情感的置信概率

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 每种情感的概率
    '''
    def predict_proba(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict_proba(samples)

    '''
    save_model(): 将模型以 model_name 命名存储在 /Models 目录下
    '''
    def save_model(self, model_name: str):
        raise NotImplementedError()

    '''
    evaluate(): 在测试集上评估模型，输出准确率

    输入:
        x_test: 样本
        y_test: 标签
    '''
    def evaluate(self, x_test, y_test):

        predictions = self.predict(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))
 
        '''
        predictions = self.predict(x_test)
        score = self.model.score(x_test, y_test)
        print("True Lable: ", y_test)
        print("Predict Lable: ", predictions)
        print("Score: ", score)
        '''

