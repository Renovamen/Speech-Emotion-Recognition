import os
import pickle
from abc import ABC
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .base import BaseModel

class MLModel(BaseModel, ABC):
    def __init__(self) -> None:
        super(MLModel, self).__init__()

    def save_model(self, config) -> None:
        """
        将模型存储在 `config.checkpoint_path` 路径下

        Args:
            config: 配置项
        """
        save_path = os.path.join(config.checkpoint_path, config.checkpoint_name + '.m')
        pickle.dump(self.model, open(save_path, "wb"))

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


class SVM(MLModel):
    def __init__(self, model_params) -> None:
        super(SVM, self).__init__()
        self.model = SVC(**model_params)


class MLP(MLModel):
    def __init__(self, model_params) -> None:
        super(MLP, self).__init__()
        self.model = MLPClassifier(**model_params)
