# Speech Emotion Recognition 

Speech emotion recognition using LSTM, SVM and MLP.

Improve the feature extraction method and get higher accuracy (about 80%). The original version is saved in [Branch: First-Version](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/First-Version).

&nbsp;

## Environment

Python 3.6.7

&nbsp;

## Structure

```
├── Common_Model.py        // Common part of all models
├── ML_Model.py            // SVM & MLP
├── DNN_Model.py           // LSTM
├── Utils.py               // Load models, plot graphs
├── Opensmile_Feature.py   // Use Opensmile for features extracting
├── Librosa_Feature.py     // Use librosa for features extracting
├── SER.py                 // Using different models for speech emotion recognition 
├── File.py                // Organize dataset (classify and rename)
├── Config.py              // Configuration parameters
├── cmd.py                 // Use argparse for getting args from command line
├── cmd_example.sh         // Examples of command line input
├── Models                 // Restore trained models
└── Feature                // Restore extracted features
```

&nbsp;

## Requirments

### Python

- [scikit-learn](https://github.com/scikit-learn/scikit-learn): SVM & MLP, split data into training set and testing set
- [Keras](https://github.com/keras-team/keras): LSTM
- [TensorFlow](https://github.com/tensorflow/tensorflow): Backend of keras
- [librosa](https://github.com/librosa/librosa): Extract features, waveform
- [SciPy](https://github.com/scipy/scipy): Spectrogram
- [pandas](https://github.com/pandas-dev/pandas): Load features
- [Matplotlib](https://github.com/matplotlib/matplotlib): Plot graphs
- [numpy](github.com/numpy/numpy)

### Tools

- [Opensmile](https://github.com/naxingyu/opensmile): Extract features

&nbsp;

## Datasets

1. [RAVDESS](https://zenodo.org/record/1188976)

   English, around 1500 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of each file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   English, around 500 audios from 4 people (male) including 7 different emotions (the first letter of each file name represents the emotional type): a = anger, d = disgust, f = fear, h = happiness, n = neutral, sa = sadness, su = surprise.

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   German, around 500 audios from 10 people (5 male and 5 female) including 7 different emotions (the second to last letter of each file name represents the emotional type): N = neutral, W = angry, A = fear, F = happy, T = sad, E = disgust, L = boredom.

4. CASIA

   Chinese, around 1200 audios from 4 people (2 male and 2 female) including 6 different emotions: neutral, happy, sad, angry, fearful and surprised.

&nbsp;

## Usage

### Prepare

Install dependencies:

```python
pip install -r requirements.txt
```

Install [Opensmile](https://github.com/naxingyu/opensmile).

&nbsp;

### Configuration

Parameters can be configured in `Config.py`.

About Opensmile standard feature sets, currently only following 6 feature sets are supported:

- `IS09_emotion`: [The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), 384 features;
- `IS10_paraling`: [The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf), 1582 features;
- `IS11_speaker_state`: [The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf), 4368 features;
- `IS12_speaker_trait`: [The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf), 6125 features;
- `IS13_ComParE`: [The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf), 6373 features;
- `ComParE_2016`: [The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), 6373 features.

You should modify `FEATURE_NUM` parameter if you need to use other feature sets.

&nbsp;

### Command Line Arguments

| Long option    | Option | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `--option`     | `-o`   | Option [ `p`: predict / `t`: train ] [ required ]            |
| `--model_type` | `-mt`  | Model type [ `svm` / `mlp` / `lstm` ] [ default is `svm` ]   |
| `--model_name` | `-mn`  | Name of the model file which will be saved or loaded [ default is `default` ] |
| `--load`       | `-l`   | Load exist features or not [ `0`: no / `1`: yes ] [ default is `1` ] |
| `--feature`    | `-f`   | How to extract features [ `o`: Opensmile / `l`: librosa ] [ default is `o` ] |
| `--audio`      | `-a`   | Path of audio which will be predicted [ default is `default.wav` ] |



Examples:

- Train:

  ```python
  python3 cmd.py -o t -mt 'svm' -mn 'SVM' -l 1 -f 'o'
  ```

- Predict:

  ```python
  python3 cmd.py -p t -mt 'svm' -mn 'SVM' -f 'o' -a [audio path]
  ```

More examples can be found in `cmd_example.sh`.

&nbsp;

### Train

The path of datasets can be configured in `Config.py`. Audios which express the same emotion should be put in the same folder (`File.py` can be used to organize the data), for example:

```
└── Datasets
    ├── Angry
    ├── Happy
    ├── Sad
    ...
```


```python
from SER import Train

'''
Input:
	model_name: model type (SVM / MLP / LSTM)
	save_model_name: name of the model file
	if_load: load exist features or not (True / False)
	feature_method: how to extract features ('o': Opensmile / 'l': librosa)
Output:
	model: a trained model
'''
model = Train(model_name, save_model_name, if_load, feature_method)
```

&nbsp;

### Load Model

```python
from Utils import load_model

'''
Input:
	load_model_name: name of the model file which will be loaded
	model_name: model type (SVM / MLP / LSTM)
Output:
	model: a model
'''
model = load_model(load_model_name, model_name)
```

&nbsp;

### Predict

```python
from SER import Predict
'''
Input:
	model: a trained or loaded model
	model_name: model type (SVM / MLP / LSTM)
	file_path: path of audio which will be predicted
	feature_method: how to extract features ('o': Opensmile / 'l': librosa)
Output:
	predict result and probability
'''
Predict(model, model_name, file_path, feature_method)
```

&nbsp;

### Extract Feature

Features extracted by Opensmile will be save in `.csv` files and by librosa will be save in `.p` files.

```python
import Librosa_Feature as of
import Opensmile_Feature as of

'''
Input:
    data_path: path of dataset / audio which will be predicted
    feature_path: path for saving features
    train: training data or not
'''

'''
Training data:
    Ouput: samples of training data, samples of testing data and their labels
'''
# Opensmile
x_train, x_test, y_train, y_test = of.get_data(data_path, feature_path, train = False)
# librosa
x_train, x_test, y_train, y_test = lf.get_data(data_path, feature_path, train = False)

'''
Predicting data:
    Output: features of audio
'''
# Opensmile
test_feature = of.get_data(data_path, feature_path, train = True)
# librosa
test_feature = lf.get_data(data_path, feature_path, train = True)
```

&nbsp;

### Load Feature

```python
import Librosa_Feature as lf
import Opensmile_Feature as of

'''
Input:
    feature_path: path for loading features
    train: training data or not
'''

'''
Training data:
    Output: samples of training data, samples of testing data and their labels
'''
# Opensmile
x_train, x_test, y_train, y_test = of.load_feature(feature_path, train = True)
# librosa
x_train, x_test, y_train, y_test = lf.load_feature(feature_path, train = True)

'''
Predicting data:
    Output: features of audio
'''
# Opensmile
test_feature = of.load_feature(feature_path, train = False)
# librosa
test_feature = lf.load_feature(feature_path, train = False)
```

&nbsp;

### Radar Chart

Plot a radar chart of probability.

Source: [Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
from Utils import Radar
'''
Input:
    data_prob: probability
'''
Radar(result_prob)
```

&nbsp;

### Play Audio

Play an audio file.

```python
from Utils import playAudio
playAudio(file_path)
```

&nbsp;

### Plot Curve

Plot loss curve or accuracy curve.

```python
from Utils import plotCurve
'''
Input:
    train(list): loss or accuracy on train set
    val(list): loss or accuracy on validation set
    title(str): title of figure
    y_label(str): label of y axis
'''
plotCurve(train, val, title, y_label)
```

&nbsp;

### Waveform

Plot a waveform of an audio.

```python
from Utils import Waveform
Waveform(file_path)
```

&nbsp;

### Spectrogram

Plot a spectrogram of an audio.

```python
from Utils import Spectrogram
Spectrogram(file_path)
```

&nbsp;

## Acknowledgements

[@Zhaofan-Su](https://github.com/Zhaofan-Su) and [@Guo Hui](https://github.com/guohui15661353950)。