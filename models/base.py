from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score

class BaseModel(ABC):
    """所有模型的基础类"""

    def __init__(self) -> None:
        self.model = None
        self.trained = False  # 模型是否已训练

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
        return self.model.predict_proba(samples)

    @abstractmethod
    def save_model(self, config) -> None:
        """保存模型"""
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        在测试集上评估模型，输出准确率

        Args:
            x_test (np.ndarray): 样本
            y_test (np.ndarray): 标签
        """
        predictions = self.predict(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy: %.3f\n' % accuracy_score(y_pred = predictions, y_true = y_test))

        """
        predictions = self.predict(x_test)
        score = self.model.score(x_test, y_test)
        print("True Lable: ", y_test)
        print("Predict Lable: ", predictions)
        print("Score: ", score)
        """
