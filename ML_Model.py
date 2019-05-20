import pickle
import sys

import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from Common_Model import Common_Model


class MLModel(Common_Model):

    def __init__(self, **params):
        super(MLModel, self).__init__(**params)

    '''
    save_model(): 将模型以 model_name 命名存储在 /Models 目录下
    '''
    def save_model(self, model_name):
        save_path = 'Models/' + model_name + '.m'
        pickle.dump(self.model, open(save_path, "wb"))

    '''
    train(): 在给定训练集上训练模型

    输入:
        x_train: 训练集样本
        y_train: 训练集标签
        x_val: 测试集样本
        y_val: 测试集标签
    '''
    def train(self, x_train, y_train, x_val = None, y_val = None):
        self.model.fit(x_train, y_train)
        self.trained = True

    '''
    predict(): 识别音频的情感

    输入:
        samples: 需要识别的音频特征

    输出:
        list: 识别结果
    '''
    def predict(self, samples):
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict(samples)

class SVM_Model(MLModel):
    def __init__(self, **params):
        params['name'] = 'SVM'
        super(SVM_Model, self).__init__(**params)
        self.model = SVC(kernel = 'rbf', probability = True, gamma = 'auto')

class MLP_Model(MLModel):
    def __init__(self, **params):
        params['name'] = 'Neural Network'
        super(MLP_Model, self).__init__(**params)
        self.model = MLPClassifier(alpha = 1.9, max_iter = 700)
