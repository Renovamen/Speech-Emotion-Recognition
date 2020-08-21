import numpy as np
import os
from utils.common import load_model, Radar, play_audio
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import utils.opts as opts


def reshape_input(model, data):
    if model == 'lstm':
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
    elif model == 'cnn1d':
        # (n_samples, n_feats) -> (n_samples, n_feats, 1)
        data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    
    return data


'''
predict(): 预测音频情感

输入:
    config(Class)
    audio_path: 要预测的音频路径
	model: 加载的模型

输出: 预测结果和置信概率
'''
def predict(config, audio_path, model):
    
    # play_audio(audio_path)

    if(config.feature_method == 'o'):
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(config, audio_path, config.predict_feature_path_opensmile, train = False)
        test_feature = of.load_feature(config, config.predict_feature_path_opensmile, train = False)
    elif(config.feature_method == 'l'):
        test_feature = lf.get_data(config, audio_path, config.predict_feature_path_librosa, train = False)
    
    test_feature = reshape_input(config.model, test_feature)
    
    result = model.predict(test_feature)
    if config.model in ['lstm', 'cnn1d', 'cnn2d']:
        result = np.argmax(result)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob, config.class_labels)


if __name__ == '__main__':

    audio_path = '/Users/zou/Desktop/Speech-Emotion-Recognition/test/angry.wav'

    config = opts.parse_opt()

    # 加载模型
    model = load_model(
        checkpoint_path = config.checkpoint_path,
        checkpoint_name = config.checkpoint_name, 
        model_name = config.model
    )

    predict(config, audio_path, model)