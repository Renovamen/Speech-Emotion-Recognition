# Speech Emotion Recognition 

用 CNN、LSTM、SVM（未完成）、MLP 进行语音情感识别。



## Environment

Python 3.6.7



## Structure

```
├── Common_Model.py        // 所有模型的通用部分（即所有模型都会继承这个类）
├── DNN_Model.py           // CNN & LSTM 模型
├── ML_Model.py            // SVM & MLP 模型
├── Utilities.py           // 读取数据 & 提取数据的特征向量
├── SER.py                 // 调用不同模型进行语音情感识别
├── File.py                // 用于整理数据集（分类、批量重命名）
├── DataSet                // 数据集                      
│   ├── Angry
│   ├── Happy
│   ...
│   ...
├── Models                 // 存储训练好的模型
```



## Requirments

- keras：LSTM & CNN
- tensorflow：keras的后端
- sklearn：SVM & MLP，划分训练集和测试集
- speechpy：提取特征向量
- librosa：读取音频
- h5py：LSTM & CNN 的模型存储在h5文件中
- numpy



## Database

1. [RAVDESS](https://zenodo.org/record/1188976)

   英文，24个人（12名男性，12名女性）的大约1500个音频，表达了8种不同的情绪（第三位数字表示情绪类别）：01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised。

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   英文，4个人（男性）的大约500个音频，表达了7种不同的情绪（第一个字母表示情绪类别）：a = anger, d = disgust, f = fear, h = happiness, n = neutral, sa = sadness, su = surprise。

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   德语，10个人的大约500个音频，表达了5种不同的情绪：happy, angry, sad, fearful, calm。

4. CASIA

   汉语，4个人（2名男性，2名女性）的大约1200个音频，表达了6种不同的情绪：neutral, happy, sad, angry, fearful, surprised。



## Usage

### Ready-made Demo

数据集放在 `/DataSet` 目录下，相同情感的音频放在同一个文件夹里（见Structure部分）。可以考虑使用 `File.py` 整理数据。



在 `SER.py` 中填入数据集路径和标签内容：

```python
DATA_PATH = 'DataSet/CASIA'
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
```

```python
from SER import LSTM
from SER import CNN
from SER import SVM
from SER import MLP

LSTM()
CNN()
SVM()
MLP()
```



### Extract Data

```python
from Utilities import get_data
x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels)
```

- x_train：训练集样本
- y_train：训练集标签
- x_test：测试集样本
- y_test：测试集标签



### Extract Feature Vector

```python
from Utilities import get_feature
# 使用 SVM & MLP 模型时要将数据降维，flatten = True
# 使用 LSTM & CNN 模型时，flatten = False
get_feature(path_of_the_audio, number_of_mfcc, flatten)
```



### Train

#### LSTM & CNN

```python
from DNN_Model import LSTM_Model
from DNN_Model import CNN_Model

model_lstm = LSTM_Model(input_shape, number_of_classes)
model_lstm.train(x_train, y_train, x_test, y_test_train, n_epochs)

model_cnn = CNN_Model(input_shape, number_of_classes)
model_cnn.train(x_train, y_train, x_test, y_test_train, n_epochs)
```



#### SVM & MLP

P.S. SVM模型还没有加进来，但如果我的预感没有错的话应该就是这样调用...

```python
from ML_Model import SVM_Model
from ML_Model import MLP_Model

model_svm = SVM_Model()
model_svm.train(x_train, y_train)

model_mlp = MLP_Model()
model_mlp.train(x_train, y_train)
```



### Evaluate Accuracy

```python
model.evaluate(x_test, y_test)
```



### Recognize

#### 训练的模型

```python
# 返回两个参数：预测结果(int)， 置信概率(numpy.ndarray)
model.recognize_one(feature_vector)
```



#### 加载的模型

```python
from Utilities import get_feature
import numpy as np
np.argmax(model.predict(np.array([get_feature(filename, flatten)])))
```



### Load Model

```python
from Utilities import load_model
# load_moedl 为模型种类(DNN / ML)
model.load_model(model_name, load_model)
```



### Save Model

模型会存储在 `/Models` 目录下。

```python
model.save_model(model_name)
```



### 雷达图

来源：[Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
from Utilities import Radar
Radar(result_prob, class_labels, num_of_classes)
```



## Acknowledgements

SVM模型和雷达图的代码来源于 [@Zhaofan-Su](https://github.com/Zhaofan-Su) 的 [SpeechEmotionRecognition](https://github.com/Zhaofan-Su/SpeechEmotionRecognition)。