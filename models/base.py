from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator

class BaseModel(ABC):
    """所有模型的基础类"""

    def __init__(
        self,
        model: Union[Sequential, BaseEstimator],
        trained: bool = False
    ) -> None:
        self.model = model
        self.trained = trained  # 模型是否已训练

    @abstractmethod
    def train(self) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        """预测音频的情感"""
        pass

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

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        """保存模型"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        """加载模型"""
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        """搭建模型"""
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        在测试集上评估模型，输出准确率

        Args:
            x_test (np.ndarray): 样本
            y_test (np.ndarray): 标签（ground truth）
        """
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        # accuracy = self.model.score(x_test, y_test)
        print('Accuracy: %.3f\n' % accuracy)

        return accuracy
