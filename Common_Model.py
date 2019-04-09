import sys
from typing import Tuple

import numpy
from sklearn.metrics import accuracy_score


class Common_Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False # 模型是否已训练

    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train (numpy.ndarray): 训练集样本
        y_train (numpy.ndarray): 训练集标签
        x_val (numpy.ndarray): 测试集样本
        y_val (numpy.ndarray): 测试集标签
        n_epochs (int): epoch

    '''
    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray, x_val: numpy.ndarray = None, y_val: numpy.ndarray = None):
        raise NotImplementedError()

    '''
    recognize(): 识别一些音频的情感

    输入:
        samples(numpy.ndarray): 需要识别的音频

    输出:
        list: 识别结果（标签）的list
    '''
    def recognize(self, samples: numpy.ndarray):
        results = []
        for _, sample in enumerate(samples):
            result, _ = self.recognize_one(sample)
            results.append(result)
        return tuple(results)

    '''
    recognize_one(): 识别某个音频的情感

    输入:
        sample: 要识别的样本
    
    返回:
        int: 识别结果
    '''
    def recognize_one(self, sample):
        raise NotImplementedError()


    '''
    save_model(): 将模型以 model_name 命名存储在 /Models 目录下
    '''
    def save_model(self, model_name: str):
        raise NotImplementedError()

    '''
    evaluate(): 在测试集上评估模型，输出准确率

    输入:
        x_test (numpy.ndarray): 样本(n维)
        y_test (numpy.ndarray): 标签(1维)
    '''
    def evaluate(self, x_test: numpy.ndarray, y_test: numpy.ndarray):
        predictions = self.recognize(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))
