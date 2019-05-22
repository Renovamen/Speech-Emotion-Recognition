from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.externals import joblib
import librosa
import librosa.display
import scipy.io.wavfile as wav

from Config import Config

'''
load_model(): 
    加载模型

输入:
    load_model_name(str): 要加载的模型的文件名
    model_name(str): 模型名称

输出:
    model: 加载好的模型
'''
def load_model(load_model_name: str, model_name: str):
    
    if(model_name == 'lstm'):
        # 加载json
        model_path = 'Models/' + load_model_name + '.h5'
        model_json_path = 'Models/' + load_model_name + '.json'
        
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # 加载权重
        model.load_weights(model_path)
    
    elif(model_name == 'svm' or model_name == 'mlp'):
        model_path = 'Models/' + load_model_name + '.m'
        model = joblib.load(model_path)

    return model


'''
Radar(): 置信概率雷达图

输入:
    data_prob(numpy.ndarray): 概率数组
'''
def Radar(data_prob):

    angles = np.linspace(0, 2 * np.pi, len(Config.CLASS_LABELS), endpoint = False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合

    fig = plt.figure()

    # polar参数
    ax = fig.add_subplot(111, polar = True)
    ax.plot(angles, data, 'bo-', linewidth=2)
    ax.fill(angles, data, facecolor='r', alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, Config.CLASS_LABELS)
    ax.set_title("Emotion Recognition", va = 'bottom')

    # 设置雷达图的数据最大值
    ax.set_rlim(0, 1)

    ax.grid(True)
    # plt.ion()
    plt.show()
    # plt.pause(4)
    # plt.close()


'''
Waveform(): 音频波形图

输入:
    file_path(str): 音频路径
'''

def Waveform(file_path: str):
    data, sampling_rate = librosa.load(file_path)
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(data, sr = sampling_rate)
    plt.show()

'''
Spectrogram(): 频谱图

输入:
    file_path(str): 音频路径
'''
def Spectrogram(file_path: str):
    # sr: 采样率
    # x: 音频数据的numpy数组
    sr, x = wav.read(file_path)

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

    plt.imshow(X.T, interpolation = 'nearest', origin = 'lower', aspect = 'auto')
    plt.show()