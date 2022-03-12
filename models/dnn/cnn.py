from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, \
    Activation, BatchNormalization, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from .dnn import DNN

class CNN1D(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(CNN1D, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        n_kernels: int,
        kernel_sizes: int,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 6,
        lr: float = 0.001
    ):
        """
        搭建模型

        Args:
            input_shape (int): 特征维度
            n_kernels (int): 卷积核数量
            kernel_sizes (list): 每个卷积层的卷积核大小，列表长度为卷积层数量
            hidden_size (int): 全连接层大小
            dropout (float, optional, default=0.5): dropout
            n_classes (int, optional, default=6): 标签种类数量
            lr (float, optional, default=0.001): 学习率
        """
        model = Sequential()

        for size in kernel_sizes:
            model.add(Conv1D(
                filters = n_kernels,
                kernel_size = size,
                padding = 'same',
                input_shape = (input_shape, 1)
            ))  # 卷积层
            model.add(BatchNormalization(axis=-1))
            model.add(Activation('relu'))
            model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(hidden_size))
        model.add(BatchNormalization(axis = -1))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))

        model.add(Dense(n_classes, activation='softmax'))  # 分类层
        optimzer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """二维数组转三维"""
        # (n_samples, n_feats) -> (n_samples, n_feats, 1)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
        return data
