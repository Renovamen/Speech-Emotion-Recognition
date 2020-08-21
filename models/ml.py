import pickle
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .common import Common_Model


class MLModel(Common_Model):

    def __init__(self, **params):
        super(MLModel, self).__init__(**params)


    '''
    save_model(): 将模型存储在 config.checkpoint_path 路径下

    输入:
        config(Class)
    '''
    def save_model(self, config):
        save_path = config.checkpoint_path + config.checkpoint_name + '.m'
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


class SVM(MLModel):
    def __init__(self, model_params, **params):
        params['name'] = 'SVM'
        super(SVM, self).__init__(**params)
        self.model = SVC(**model_params)


class MLP(MLModel):
    def __init__(self, model_params, **params):
        params['name'] = 'Neural Network'
        super(MLP, self).__init__(**params)
        self.model = MLPClassifier(**model_params)