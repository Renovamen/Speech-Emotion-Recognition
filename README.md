# Speech Emotion Recognition 

用 SVM、MLP 进行语音情感识别。

&nbsp;

## Environment

Python 3.6.7

&nbsp;

## Structure

```
├── Common_Model.py        // 所有模型的通用部分（即所有模型都会继承这个类）
├── ML_Model.py            // SVM & MLP 模型
├── Utilities.py           // 提取数据的特征向量和绘图等
├── SER.py                 // 调用不同模型进行语音情感识别
├── File.py                // 用于整理数据集（分类、批量重命名）
├── DataSet                // 数据集                      
│   ├── Angry
│   ├── Happy
│   ...
│   ...
├── Models                 // 存储训练好的模型
└── Feature                // 存储提取好的特征
```

&nbsp;

## Requirments

### Python

- [scikit-learn](https://github.com/scikit-learn/scikit-learn)：SVM & MLP 模型，划分训练集和测试集
- [librosa](https://github.com/librosa/librosa)：波形图
- [SciPy](https://github.com/scipy/scipy)：频谱图

- [pandas](https://github.com/pandas-dev/pandas)：加载特征
- [Matplotlib](https://github.com/matplotlib/matplotlib)：画图
- [numpy](github.com/numpy/numpy)

### Tools

- [Opensmile](https://github.com/naxingyu/opensmile)：提取特征

&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   英文，24 个人（12 名男性，12 名女性）的大约 1500 个音频，表达了 8 种不同的情绪（第三位数字表示情绪类别）：01 = neutral，02 = calm，03 = happy，04 = sad，05 = angry，06 = fearful，07 = disgust，08 = surprised。

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   英文，4 个人（男性）的大约 500 个音频，表达了 7 种不同的情绪（第一个字母表示情绪类别）：a = anger，d = disgust，f = fear，h = happiness，n = neutral，sa = sadness，su = surprise。

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，10 个人（5 名男性，5 名女性）的大约 500 个音频，表达了 7 种不同的情绪：nertral，anger，fear，joy，sadness，disgust，boredom。

4. CASIA

   汉语，4 个人（2 名男性，2 名女性）的大约 1200 个音频，表达了 6 种不同的情绪：neutral，happy，sad，angry，fearful，surprised。

&nbsp;

## Usage

### Prepare

安装依赖：

```python
pip install -r requirements.txt
```

安装 [Opensmile](https://github.com/naxingyu/opensmile)。



### Train

数据集放在 `/DataSet` 目录下，相同情感的音频放在同一个文件夹里（见 Structure 部分）。可以考虑使用 `File.py` 整理数据。

在 `SER.py` 中填入数据集路径 `DATA_PATH` 和标签名称 `CLASS_LABELS`，如：


```python
from ML_Model import SVM_Model
from ML_Model import MLP_Model

'''
输入:
	model_name: 模型名称(SVM / MLP)
	save_model_name: 保存模型的文件名
输出：
	model: 训练好的模型
'''
model = Train(model_name, save_model_name)
```



### Load Model

```python
from Utilities import load_model

'''
输入:
	model_name: 模型名称（SVM / MLP）
	load_model: 模型种类（ML / DNN）
输出：
	model: 训练好的模型
'''
model = load_model(model_name, load_model)
```



### Predict

```python
'''
输入:
	model: 已加载或训练的模型，
	save_model_name: 保存模型的文件名
输出：
	file_path: 要预测的文件路径
'''
Predict(model, file_path)
```



### Extract Feature

```python
from Utilities import get_data

'''
输入:
    opensmile_path: Opensmile 安装路径
    data_path: 数据集文件夹路径
    feature_path: 保存特征的路径
    config: Opensmile 配置文件（要提取哪些特征）
    class_labels: 标签
    train: 是否为训练数据
'''

# 训练数据
'''
输出: 训练数据、测试数据特征和对应的标签
'''
x_train, x_test, y_train, y_test = get_data(opensmile_path, data_path, feature_path, cofig, class_labels, train = False)

# 预测数据
'''
输出: 预测数据特征
'''
test_feature = get_data(opensmile_path, data_path, feature_path, cofig, class_labels, train = True)
```



### Load Feature

```python
from Utilities import load_feature

'''
从 csv 加载特征数据

输入:
    feature_path: 特征文件路径
    train: 是否为训练数据
'''

# 训练数据
'''
输出: 训练数据、测试数据和对应的标签
'''
x_train, x_test, y_train, y_test = load_feature(feature_path, train = True)

# 预测数据
'''
输出: 预测数据特征
'''
test_feature = load_feature(feature_path, train = False)
```



### Radar Chart

画出置信概率的雷达图。

来源：[Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
from Utilities import Radar
'''
输入:
    data_prob: 概率数组
    class_labels: 标签
'''
Radar(result_prob, class_labels)
```



### Waveform

画出音频的波形图。

```python
from Utilities import Waveform
Waveform(file_path)
```



### Spectrogram

画出音频的频谱图。

```python
from Utilities import Spectrogram
Spectrogram(file_path)
```



## Acknowledgements

[@Zhaofan-Su](https://github.com/Zhaofan-Su) 和 [@Guo Hui](https://github.com/guohui15661353950)。