import numpy as np
import os
from misc.utils import load_model, Radar, play_audio
import extract_feats.opensmile as of
import extract_feats.librosa as lf
from config import config
import misc.opts as opts

'''
predict(): 预测音频情感

输入:
	model: 已加载或训练的模型
	model_name: 模型名称
	file_path: 要预测的文件路径
    feature_method: 提取特征的方法（'o': Opensmile / 'l': librosa）

输出: 预测结果和置信概率
'''
def predict(model, model_name: str, file_path: str, feature_method: str = 'o'):
    
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_path
    play_audio(file_path)

    if(feature_method == 'o'):
        # 一个玄学 bug 的暂时性解决方案
        of.get_data(file_path, config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
        test_feature = of.load_feature(config.PREDICT_FEATURE_PATH_OPENSMILE, train = False)
    elif(feature_method == 'l'):
        test_feature = lf.get_data(file_path, config.PREDICT_FEATURE_PATH_LIBROSA, train = False)
    
    if(model_name == 'lstm'):
        # 二维数组转三维（samples, time_steps, input_dim）
        test_feature = np.reshape(test_feature, (test_feature.shape[0], 1, test_feature.shape[1]))
    
    result = model.predict(test_feature)
    if(model_name == 'lstm'):
        result = np.argmax(result)

    result_prob = model.predict_proba(test_feature)[0]
    print('Recogntion: ', config.CLASS_LABELS[int(result)])
    print('Probability: ', result_prob)
    Radar(result_prob)


if __name__ == '__main__':

    opt = opts.parse_pred()

    # 加载模型
    model = load_model(load_model_name = opt.model_name, model_name = opt.model_type)
    predict(model, model_name = opt.model_type, file_path = opt.audio, feature_method = opt.feature)

    # model = load_model(load_model_name = "LSTM_OPENSMILE", model_name = "lstm")
    # predict(model, model_name = "lstm", file_path = 'test/angry.wav', feature_method = 'l')