from .dnn import LSTM
from .dnn import CNN1D
from .ml import SVM
from .ml import MLP

'''
setup(): 创建模型

输入:
    config(Class)
    n_feats(int): 特征数量（神经网络输入张量大小）
'''
def setup(config, n_feats):

    if config.model == 'svm':
        model = SVM(model_params = config.params)
    elif config.model == 'mlp':
        model = MLP(model_params = config.params)
    elif config.model == 'lstm':
        model = LSTM(
            input_shape = n_feats, 
            num_classes = len(config.class_labels),
            lr = config.lr,
            rnn_size = config.rnn_size,
            hidden_size = config.hidden_size,
            dropout = config.dropout
        )
    elif config.model == 'cnn1d':
        model = CNN1D(
            input_shape = n_feats, 
            num_classes = len(config.class_labels),
            lr = config.lr,
            n_kernels = config.n_kernels,
            kernel_sizes = config.kernel_sizes,
            hidden_size = config.hidden_size,
            dropout = config.dropout
        )

    return model