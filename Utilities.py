# 从给定文件夹读取数据和提取MFCC特征

import os
import sys
from typing import Tuple
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
from speechpy.feature import mfcc
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.externals import joblib
import matplotlib.pyplot as plt

mean_signal_length = 48000

"""
get_feature(): 提取某个音频的MFCC特征向量

输入:
    file_path(str): 该音频路径
    mfcc_len(int): 每帧的MFCC特征数
    flatten(bool): 是否降维数据

输出:
    numpy.ndarray: 该音频的MFCC特征向量
"""
def get_feature(file_path: str, mfcc_len: int = 39, flatten: bool = False):
    # 某些音频用scipy.io.wavfile读会报 "Incomplete wav chunk" error
    # 似乎是因为scipy只能读pcm和float格式，而有的wav不是这两种格式...
    # fs, signal = wav.read(file_path)
    signal, fs = librosa.load(file_path)


    s_len = len(signal)

    # 如果音频信号小于mean_signal_length，则扩充它
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    
    # 否则把它切开
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]

    mel_coefficients = mfcc(signal, fs, num_cepstral = mfcc_len)
    #  用 SVM & MLP 模型时要降维数据
    if flatten:
        mel_coefficients = np.ravel(mel_coefficients)

    return mel_coefficients

def get_feature_svm(file_path: str, mfcc_len: int = 48):
    y, sr = librosa.load(file_path)

    # 对于每一个音频文件提取其mfcc特征
    # y:音频时间序列;
    # n_mfcc:要返回的MFCC数量
    mfcc_feature = librosa.feature.mfcc(y, sr, n_mfcc = 48)
    zcr_feature = librosa.feature.zero_crossing_rate(y)
    energy_feature = librosa.feature.rmse(y)
    rms_feature = librosa.feature.rmse(y)

    mfcc_feature = mfcc_feature.T.flatten()[:mfcc_len]
    zcr_feature = zcr_feature.flatten()
    energy_feature = energy_feature.flatten()
    rms_feature = rms_feature.flatten()

    zcr_feature = np.array([np.mean(zcr_feature)])
    energy_feature = np.array([np.mean(energy_feature)])
    rms_feature = np.array([np.mean(rms_feature)])

    data_feature = np.concatenate((mfcc_feature, zcr_feature, energy_feature, rms_feature))
    return data_feature

'''
get_data(): 
    提取数据（训练集和测试集）: 遍历所有文件夹, 读取每个文件夹中的音频, 提取每个音频的MFCC特征向量

输入:
    data_path(str): 数据集文件夹路径
    mfcc_len(int): 每帧的MFCC特征数
    class_labels(tuple): 标签
    flatten(bool): 是否降维数据

输出:
    训练集和测试集的MFCC数组和对应的labels数组(numpy.ndarray)
    标签数量(int)
'''
def get_data(data_path: str, mfcc_len: int = 39, class_labels: Tuple = ("angry", "fear", "happy", "neutral", "sad", "surprise"), flatten: bool = False, _svm: bool = False):
    data = []
    labels = []
    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    # 遍历文件夹
    for i, directory in enumerate(class_labels):
        sys.stderr.write("Started reading folder %s\n" % directory)
        os.chdir(directory)
        # 读取该文件夹下的音频
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.getcwd() + '/' + filename
            # 提取该音频的特征向量
            if _svm:
                feature_vector = get_feature_svm(file_path = filepath, mfcc_len = mfcc_len)
            else:
                feature_vector = get_feature(file_path = filepath, mfcc_len = mfcc_len, flatten = flatten)
            data.append(feature_vector)
            labels.append(i)
        sys.stderr.write("Ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size = 0.2, random_state = 42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


'''
load_model_dnn(): 
    加载 CNN & LSTM 的模型

输入:
    model_name(str): 模型名称
    load_model(str): 模型种类（DNN / ML）

输出:
    model: 加载好的模型
'''
def load_model(model_name: str, load_model: str):
    
    if load_model == 'DNN':
        # 加载json
        model_path = 'Models/' + model_name + '.h5'
        model_json_path = 'Models/' + model_name + '.json'
        
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # 加载权重
        model.load_weights(model_path)
    
    elif load_model == 'ML':
        model_path = 'Models/' + model_name + '.m'
        model = joblib.load(model_path)

    return model


'''
plotLine(): 
    绘制损失率和准确率的折线图

输入:
    train(list): 训练集损失率或准确率数组
    val(list): 测试集损失率或准确率数组
    title(str): 图像标题
    y_label(str): y 轴标题
'''
def plotLine(train, val, title: str, y_label: str):
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


'''
Radar(): 
    置信概率雷达图

输入:
    data_prob(numpy.ndarray): 概率数组
    class_labels(tuple): 标签
    num_classes(int): 标签数量
'''
def Radar(data_prob, class_labels: Tuple, num_classes: int):

    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint = False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    fig = plt.figure()

    # polar参数
    ax = fig.add_subplot(111, polar = True)
    ax.plot(angles, data, 'bo-', linewidth=2)
    ax.fill(angles, data, facecolor='r', alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va = 'bottom')

    # 设置雷达图的数据最大值
    ax.set_rlim(0, 1)

    ax.grid(True)
    # plt.ion()
    plt.show()
    # plt.pause(4)
    # plt.close()


'''
Waveform(): 
    音频波形图

输入:
    file_path(str): 音频路径
'''

def Waveform(file_path: str):
    data, sampling_rate = librosa.load(file_path)
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr=sampling_rate)
    plt.show()

'''
Spectrogram(): 
    频谱图

输入:
    file_path(str): 音频路径
'''
def Spectrogram(file_path: str):
    # sr: 采样率
    # x: 音频数据的numpy数组
    sr,x = wav.read(file_path)

    # step: 10ms, window: 30ms
    nstep = int(sr * 0.01)
    nwin  = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)

    nn = range(nwin, len(x), nstep)
    X = np.zeros( (len(nn), nfft//2) )

    for i,n in enumerate(nn):
        xseg = x[n-nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i,:] = np.log(np.abs(z[:nfft//2]))

    plt.imshow(X.T, interpolation='nearest', origin='lower', aspect='auto')
    plt.show()