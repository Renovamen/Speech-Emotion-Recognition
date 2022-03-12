import os
import re
import sys
import librosa
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import utils

def features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    # fmin 和 fmax 对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # 频谱质心
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # 谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # 使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features

def extract_features(file: str, pad: bool = False) -> np.ndarray:
    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)

def get_max_min(files: list) -> Tuple[float]:
    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_

def get_data_path(data_path: str, class_labels: list) -> list:
    """
    获取所有音频的路径

    Args:
        data_path (str): 数据集文件夹路径
        class_labels (list): 情感标签
    Returns:
        wav_file_path (list): 所有音频的路径
    """
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    # 遍历文件夹
    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        # 读取该文件夹下的音频
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    从 "{config.feature_folder}/*.p" 文件中加载特征数据

    Args:
        config: 配置项
        train (bool): 是否为训练数据

    Returns:
        - X (Tuple[np.ndarray]): 训练特征、测试特征和对应的标签
        - X (np.ndarray): 预测特征
    """
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")

    features = pd.DataFrame(
        data = joblib.load(feature_path),
        columns = ['file_name', 'features', 'emotion']
    )

    X = list(features['features'])
    Y = list(features['emotion'])

    # 标准化模型路径
    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_LIBROSA.m')

    if train == True:
        # 标准化数据
        scaler = StandardScaler().fit(X)
        # 保存标准化模型
        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    else:
        # 标准化数据
        # 加载标准化模型
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    提取所有音频的特征: 遍历所有文件夹, 读取每个文件夹中的音频, 提取每个音频的特征，把所有特征
    保存在 "{config.feature_folder}/*.p" 文件中。

    Args:
        config: 配置项
        data_path (str): 数据集文件夹/测试文件路径
        train (bool): 是否为训练数据

    Returns:
        - train = True: 训练特征、测试特征和对应的标签
        - train = False: 预测特征
    """
    if train == True:
        files = get_data_path(data_path, config.class_labels)
        max_, min_ = get_max_min(files)

        mfcc_data = []
        for file in files:
            label = re.findall(".*-(.*)-.*", file)[0]

            # 三分类
            # if(label == "sad" or label == "neutral"):
            #     label = "neutral"
            # elif(label == "angry" or label == "fear"):
            #     label = "negative"
            # elif(label == "happy" or label == "surprise"):
            #     label = "positive"

            features = extract_features(file, max_)
            mfcc_data.append([file, features, config.class_labels.index(label)])

    else:
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    # 如果 config.feature_folder 文件夹不存在，则新建一个
    utils.mkdirs(config.feature_folder)
    # 特征存储路径
    feature_path = os.path.join(config.feature_folder, "train.p" if train == True else "predict.p")
    # 保存特征
    pickle.dump(mfcc_data, open(feature_path, 'wb'))

    return load_feature(config, train=train)
