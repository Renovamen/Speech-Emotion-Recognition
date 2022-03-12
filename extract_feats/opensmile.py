import os
import csv
import sys
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import utils

# 每个特征集的特征数量
FEATURE_NUM = {
    'IS09_emotion': 384,
    'IS10_paraling': 1582,
    'IS11_speaker_state': 4368,
    'IS12_speaker_trait': 6125,
    'IS13_ComParE': 6373,
    'ComParE_2016': 6373
}

def get_feature_opensmile(config, filepath: str) -> list:
    """
    用 Opensmile 提取一个音频的特征

    Args:
        config: 配置项
        file_path (str): 音频路径

    Returns:
        vector (list): 该音频的特征向量
    """

    # 项目路径
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # single_feature.csv 路径
    single_feat_path = os.path.join(BASE_DIR, config.feature_folder, 'single_feature.csv')
    # Opensmile 配置文件路径
    opensmile_config_path = os.path.join(config.opensmile_path, 'config', config.opensmile_config + '.conf')

    # Opensmile 命令
    cmd = 'cd ' + config.opensmile_path + ' && ./SMILExtract -C ' + opensmile_config_path + ' -I ' + filepath + ' -O ' + single_feat_path + ' -appendarff 0'
    print("Opensmile cmd: ", cmd)
    os.system(cmd)

    reader = csv.reader(open(single_feat_path,'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1: FEATURE_NUM[config.opensmile_config] + 1]

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    从 "{config.feature_folder}/*.csv" 文件中加载特征数据

    Args:
        config: 配置项
        train (bool): 是否为训练数据

    Returns:
        - X (Tuple[np.ndarray]): 训练特征、测试特征和对应的标签
        - X (np.ndarray): 预测特征
    """

    feature_path = os.path.join(config.feature_folder, "train.csv" if train == True else "predict.csv")

    # 加载特征数据
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, FEATURE_NUM[config.opensmile_config] + 1)]

    X = df.loc[:,features].values
    Y = df.loc[:,'label'].values

    # 标准化模型路径
    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_OPENSMILE.m')

    if train == True:
        # 标准化数据
        scaler = StandardScaler().fit(X)
        # 保存标准化模型
        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)
        # 划分训练集和测试集
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test
    else:
        # 标准化数据
        # 加载标准化模型
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    用 Opensmile 提取所有音频的特征: 遍历所有文件夹, 读取每个文件夹中的音频, 提取每个音频的
    特征，把所有特征保存在 "{config.feature_path}/*.csv" 文件中。

    Args:
        config: 配置项
        data_path (str): 数据集文件夹 / 测试文件路径
        train (bool): 是否为训练数据

    Returns:
        - train = True: 训练特征、测试特征和对应的标签
        - train = False: 预测特征
    """
    # 如果 config.feature_folder 文件夹不存在，则新建一个
    utils.mkdirs(config.feature_folder)

    # 特征存储路径
    feature_path = os.path.join(config.feature_folder, "train.csv" if train == True else "predict.csv")

    # 写表头
    writer = csv.writer(open(feature_path, 'w'))
    first_row = ['label']
    for i in range(1, FEATURE_NUM[config.opensmile_config] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, 'a+'))
    print('Opensmile extracting...')

    if train == True:
        cur_dir = os.getcwd()
        sys.stderr.write('Curdir: %s\n' % cur_dir)
        os.chdir(data_path)

        # 遍历文件夹
        for i, directory in enumerate(config.class_labels):
            sys.stderr.write("Started reading folder %s\n" % directory)
            os.chdir(directory)

            # label_name = directory
            label = config.class_labels.index(directory)

            # 读取该文件夹下的音频
            for filename in os.listdir('.'):
                if not filename.endswith('wav'):
                    continue
                filepath = os.path.join(os.getcwd(), filename)

                # 提取该音频的特征
                feature_vector = get_feature_opensmile(config, filepath)
                feature_vector.insert(0, label)
                # 把每个音频的特征整理到一个 csv 文件中
                writer.writerow(feature_vector)

            sys.stderr.write("Ended reading folder %s\n" % directory)
            os.chdir('..')
        os.chdir(cur_dir)

    else:
        feature_vector = get_feature_opensmile(config, data_path)
        feature_vector.insert(0, '-1')
        writer.writerow(feature_vector)

    print('Opensmile extract done.')

    # 一个玄学 bug 的暂时性解决方案
    # 这里无法直接加载除了 IS10_paraling 以外的其他特征集的预测数据特征，非常玄学
    if train == True:
        return load_feature(config, train=train)
