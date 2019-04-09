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
        x_train (numpy.ndarray): 训练集样本
        y_train (numpy.ndarray): 训练集标签
        x_val (numpy.ndarray): 测试集样本
        y_val (numpy.ndarray): 测试集标签
    '''
    def train(self, x_train, y_train, x_val = None, y_val = None):
        self.model.fit(x_train, y_train)
        self.trained = True


    '''
    recognize_one(): 识别某个音频的情感

    输入:
        sample: 要预测的样本
    '''    
    def recognize_one(self, sample):
        # 没有训练和加载过模型
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return self.model.predict(numpy.array([sample])), self.model.predict_proba(numpy.array([sample]))[0]


class SVM_Model(MLModel):
    def __init__(self, **params):
        '''
        C: 误差项惩罚参数，对误差的容忍程度。C越大，越不能容忍误差
        gamma: 选择RBF函数作为kernel，越大，支持的向量越少；越小，支持的向量越多
        '''
        params['name'] = 'SVM'
        super(SVM_Model, self).__init__(**params)
        self.model = SVC(decision_function_shape='ovo', kernel = 'rbf', C = 10, gamma = 0.0001, probability = True)

class MLP_Model(MLModel):
    def __init__(self, **params):
        params['name'] = 'Neural Network'
        super(MLP_Model, self).__init__(**params)
        self.model = MLPClassifier(activation = 'logistic', verbose = True, hidden_layer_sizes = (512,), batch_size = 32)
