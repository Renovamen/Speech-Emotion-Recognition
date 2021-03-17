import os
from typing import Tuple, Optional
import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense
from ..base import BaseModel
from utils.common import plotCurve

class DNN(BaseModel):
    """
    所有基于 Keras 的深度学习模型的基本类

    Args:
        input_shape (Tuple[int]): 特征维度
        num_classes (int): 标签种类数量
        lr (float): 学习率
    """
    def __init__(
        self, input_shape: Tuple[int], num_classes: int, lr: float, **params
    ) -> None:
        super(DNN, self).__init__()

        self.input_shape = input_shape

        self.model = Sequential()
        self.make_model(**params)
        self.model.add(Dense(num_classes, activation = 'softmax'))

        optimzer = keras.optimizers.Adam(lr = lr)
        self.model.compile(loss = 'categorical_crossentropy', optimizer = optimzer, metrics = ['accuracy'])

        print(self.model.summary())

    def save_model(self, config) -> None:
        """
        将模型存储在 `config.checkpoint_path` 路径下

        Args:
            config: 配置项
        """
        h5_save_path = os.path.join(config.checkpoint_path, config.checkpoint_name + '.h5')
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(config.checkpoint_path, config.checkpoint_name + '.json')
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    def reshape_input(self):
        NotImplementedError()

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        n_epochs: int = 50
    ) -> None:
        """
        在给定训练集上训练模型

        Args:
            x_train (np.ndarray): 训练集样本
            y_train (np.ndarray): 训练集标签
            x_val (np.ndarray, optional): 测试集样本
            y_val (np.ndarray, optional): 测试集标签
            batch_size (int): 批大小
            n_epochs (int): epoch 数
        """
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        history = self.model.fit(
            x_train, y_train,
            batch_size = batch_size,
            epochs = n_epochs,
            shuffle = True, # 每个 epoch 开始前随机排列训练数据
            validation_data = (x_val, y_val)
        )

        # 训练集上的损失和准确率
        acc = history.history['acc']
        loss = history.history['loss']
        # 验证集上的损失和准确率
        val_acc = history.history['val_acc']
        val_loss = history.history['val_loss']

        plotCurve(acc, val_acc, 'Accuracy', 'acc')
        plotCurve(loss, val_loss, 'Loss', 'loss')

        self.trained = True

    def predict(self, sample: np.ndarray) -> np.ndarray:
        """
        识别音频的情感

        Args:
            samples (np.ndarray): 需要识别的音频特征

        Returns:
            results (np.ndarray): 识别结果
        """
        sample = self.reshape_input(sample)

        # 没有训练和加载过模型
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        return np.argmax(self.model.predict(sample), axis=1)

    def make_model(self):
        raise NotImplementedError()
