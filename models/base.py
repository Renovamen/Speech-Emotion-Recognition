import numpy as np
from sklearn.metrics import accuracy_score

class BaseModel:
    """所有模型的基础类"""

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        self.model = None
        self.trained = False  # 模型是否已训练

    def train(self, x_train, y_train, x_val, y_val) -> None:
        """
        在给定训练集上训练模型

        Args:
            x_train (np.ndarray): 训练集样本
            y_train (np.ndarray): 训练集标签
            x_val (np.ndarray): 测试集样本
            y_val (np.ndarray): 测试集标签
        """
        raise NotImplementedError()

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        识别音频的情感

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 识别结果
        """
        raise NotImplementedError()

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """
        音频情感的置信概率

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 每种情感的概率
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')
        return self.model.predict_proba(samples)

    def save_model(self, model_name: str):
        """保存模型"""
        raise NotImplementedError()

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
