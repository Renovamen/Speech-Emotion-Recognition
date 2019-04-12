# Speech Emotion Recognition 

Speech emotion recognition using CNN, LSTM, SVM and MLP.



## Environment

Python 3.6.7



## Structure

```
├── Common_Model.py        // Common part of all models
├── DNN_Model.py           // CNN & LSTM
├── ML_Model.py            // SVM & MLP
├── Utilities.py           // Load data & extract feature vectors
├── SER.py                 // Using different models for speech emotion recognition 
├── File.py                // Organize dataset (classify and rename)
├── DataSet                // Dataset folder                      
│   ├── Angry
│   ├── Happy
│   ...
│   ...
├── Models                 // A folder which restore trained models
```



## Requirments

- keras：LSTM & CNN
- tensorflow：backend of keras
- sklearn：SVM & MLP, divide data into training set and testing set
- speechpy：extract feature vectors
- librosa：extract audio
- h5py：save trained models of LSTM & CNN in h5 files
- numpy



## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   English, around 1500 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of each file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   English, around 500 audios from 4 people (male) including 7 different emotions (the first letter of each file name represents the emotional type): a = anger, d = disgust, f = fear, h = happiness, n = neutral, sa = sadness, su = surprise.

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   German, around 500 audios from 10 people including 5 different emotions: happy, angry, sad, fearful and calm.

4. CASIA

   Chinese, around 1200 audios from 4 people (2 male and 2 female) including 6 different emotions: neutral, happy, sad, angry, fearful and surprised.



## Usage

### Ready-made Demo

Dataset should be put in  `/DataSet` directory and audios which express the same emotion should be put in the same folder (the Structure section has given an example).  `File.py` can be used to organize the data.

Put the path of dataset `DATA_PATH` and the names of labels `CLASS_LABELS` in `SER.py`, for example:

```python
DATA_PATH = 'DataSet/CASIA'
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")
```

```python
from SER import LSTM
from SER import CNN
from SER import SVM
from SER import MLP

# file_path: the path of the test audio
LSTM(file_path)
CNN(file_path)
SVM(file_path)
MLP(file_path)
```



### Extract Data

```python
from Utilities import get_data
# When using SVM, _svm = True, or _svm = False
x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels)
```

- x_train：samples of training data
- y_train：labels of training data
- x_test：samples of testing data
- y_test：labels of testing data



### Extract Feature Vector

```python
from Utilities import get_feature
# Data should be flattened when using SVM, flatten = True
# When usin LSTM & CNN, flatten = False
get_feature(path_of_the_audio, number_of_mfcc, flatten)

# Get features when using SVM
get_feature_svm(path_of_the_audio, number_of_mfcc)
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

#### Trained model

```python
# return two parameters: result(int), confidence probability(numpy.ndarray)
model.recognize_one(feature_vector)
```



#### Loaded model

```python
from Utilities import get_feature
import numpy as np
np.argmax(model.predict(np.array([get_feature(filename, flatten)])))
```



### Load Model

```python
from Utilities import load_model
# load_model: the type of model (DNN / ML)
model.load_model(model_name, load_model)
```



### Save Model

Model will be saves in `/Models` directory.

```python
model.save_model(model_name)
```



### Radar Chart

Plot a radar chart of confidence probability.

Source：[Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
from Utilities import Radar
Radar(result_prob, class_labels, num_of_classes)
```



### Waveform

Plot a waveform of an audio.

```python
from Utilities import Waveform
Waveform(path_of_audio)
```



### Spectrogram

Plot a spectrogram of an audio.

```python
from Utilities import Spectrogram
Spectrogram(path_of_audio)
```



## Acknowledgements

The codes of SVM model and radar chart are from [SpeechEmotionRecognition](https://github.com/Zhaofan-Su/SpeechEmotionRecognition) of [@Zhaofan-Su](https://github.com/Zhaofan-Su) and [@Guo Hui](https://github.com/guohui15661353950).

