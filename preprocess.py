"""提取数据集中音频的特征并保存"""

import extract_feats.opensmile as of
import extract_feats.librosa as lf
from utils import parse_opt

if __name__ == '__main__':
    config = parse_opt()

    if config.feature_method == 'o':
        of.get_data(config, config.data_path, train=True)

    elif config.feature_method == 'l':
        lf.get_data(config, config.data_path, train=True)
