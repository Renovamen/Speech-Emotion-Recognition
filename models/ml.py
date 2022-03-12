import os
import pickle
from abc import ABC
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
import joblib
from .base import BaseModel

class MLModel(BaseModel, ABC):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLModel, self).__init__(model, trained)

    def save(self, path: str, name: str) -> None:
        """
        保存模型

        Args:
            path (str): 模型路径
            name (str): 模型文件名
        """
        save_path = os.path.abspath(os.path.join(path, name + '.m'))
        pickle.dump(self.model, open(save_path, "wb"))

    @classmethod
    def load(cls, path: str, name: str):
        """
        加载模型

        Args:
            path (str): 模型路径
            name (str): 模型文件名
        """
        model_path = os.path.abspath(os.path.join(path, name + '.m'))
        model = joblib.load(model_path)
        return cls(model, True)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        训练模型

        Args:
            x_train (np.ndarray): 训练集样本
            y_train (np.ndarray): 训练集标签
        """
        self.model.fit(x_train, y_train)
        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        预测音频的情感

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 识别结果
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')
        return self.model.predict(samples)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """
        预测音频的情感的置信概率

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 每种情感的概率
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        if hasattr(self, 'reshape_input'):
            samples = self.reshape_input(samples)
        return self.model.predict_proba(samples)[0]


class SVM(MLModel):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(SVM, self).__init__(model, trained)

    @classmethod
    def make(cls, params):
        model = SVC(**params)
        return cls(model)


class MLP(MLModel):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLP, self).__init__(model, trained)

    @classmethod
    def make(cls, params):
        model = MLPClassifier(**params)
        return cls(model)
