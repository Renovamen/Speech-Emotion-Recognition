'''
提取数据集中音频的特征并保存
'''

import extract_feats.opensmile as of
import extract_feats.librosa as lf
from config import config
import misc.opts as opts


if __name__ == '__main__':

    opt = opts.parse_prepro()
    feat_method = opt.feature
    # feat_method = 'o'
    
    if(feat_method == 'o'):
        of.get_data(config.DATA_PATH, config.TRAIN_FEATURE_PATH_OPENSMILE, train = True)

    elif(feat_method == 'l'):
        lf.get_data(config.DATA_PATH, config.TRAIN_FEATURE_PATH_LIBROSA, train = True)