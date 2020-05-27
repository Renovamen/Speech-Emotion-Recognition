# Speech Emotion Recognition 

Speech emotion recognition using LSTM, SVM and MLP, implemented in Keras.

We have improved the feature extracting method and achieved higher accuracy (about 80%). The original version is backed up under [First-Version](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/First-Version) branch.

English Document | [中文文档](README.md)

&nbsp;

## Environment

Python 3.6.7

Keras 2.2.4

&nbsp;

## Structure

```
├── models/                // model implementations
│   ├── common.py          // common part of all models
│   ├── dnn.py             // LSTM
│   └── ml.py              // SVM & MLP
├── extract_feats/
│   ├── librosa.py         // extract features using librosa
│   └── opensmile.py       // extract features using Opensmile
├── misc/
│   ├── files.py           // setup dataset (classify and rename)
│   ├── opts.py            // use argparse to get args from command line
│   └── utils.py           // load models, plot graphs
├── features/              // store extracted features
├── config.py              // configure parameters
├── train.py               // train
├── predict.py             // recognize the emotion of a given audio
├── preprocess.py          // data preprocessing (extract features and store them locally)
└── example.sh             // examples of command line inputs
```

&nbsp;

## Requirments

### Python

- [scikit-learn](https://github.com/scikit-learn/scikit-learn): SVM & MLP, split data into training set and testing set
- [Keras](https://github.com/keras-team/keras): LSTM
- [TensorFlow](https://github.com/tensorflow/tensorflow): Backend of Keras
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

   English, around 1500 audios from 24 people (12 male and 12 female) including 8 different emotions (the third number of the file name represents the emotional type): 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised.

2. [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

   English, around 500 audios from 4 people (male) including 7 different emotions (the first letter of the file name represents the emotional type): a = anger, d = disgust, f = fear, h = happiness, n = neutral, sa = sadness, su = surprise.

3. [EMO-DB](http://www.emodb.bilderbar.info/download/)

   German, around 500 audios from 10 people (5 male and 5 female) including 7 different emotions (the second to last letter of the file name represents the emotional type): N = neutral, W = angry, A = fear, F = happy, T = sad, E = disgust, L = boredom.

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

Parameters can be configured in [`config.py`](config.py).

It should be noted that, currently only the following 6 Opensmile standard feature sets are supported:

- `IS09_emotion`: [The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), 384 features;
- `IS10_paraling`: [The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf), 1582 features;
- `IS11_speaker_state`: [The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf), 4368 features;
- `IS12_speaker_trait`: [The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf), 6125 features;
- `IS13_ComParE`: [The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf), 6373 features;
- `ComParE_2016`: [The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), 6373 features.

You may should modify item  `FEATURE_NUM` if you want to use other feature sets.

&nbsp;

### Preprocess

First of all, you should extract features of each audio in dataset and store them locally. Features extracted by Opensmile will be saved in `.csv` files and by librosa will be saved in `.p` files.

| Long option | Option | Description                                                  |
| ----------- | ------ | ------------------------------------------------------------ |
| `--feature` | `-f`   | ow to extract features [ `o`: Opensmile / `l`: librosa ] [ default is `o` ] |

Example:

```python
python preprocess.py -f 'o'
```

More examples can be found in [`example.sh`](example.sh).

&nbsp;

### Train

The path of the datasets can be configured in [`config.py`](config.py). Audios which express the same emotion should be put in the same folder (you may want to refer to [`misc/files.py`](misc/files.py) when setting up datasets), for example:

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

&nbsp;

Argparse：

| Long option    | Option | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `--model_type` | `-mt`  | model type [ `svm` / `mlp` / `lstm` ] [ default is `svm` ]   |
| `--model_name` | `-mn`  | name of the model file that will be saved [ default is `default` ] |
| `--feature`    | `-f`   | how to extract features [ `o`: Opensmile / `l`: librosa ] [ default is `o` ] |


Example：

```python
python train.py -mt 'svm' -mn 'SVM' -f 'o'
```

More examples can be found in [`example.sh`](example.sh).

&nbsp;

If you don't want to set parameters via command line:

```python
from train import train

'''
input params:
	model_name: model type (svm / mlp / lstm)
	save_model_name: name of the model file
	feature_method: how to extract features ('o': Opensmile / 'l': librosa)
'''
train(model_name = "lstm", save_model_name = "LSTM", feature_method = 'l')
```

&nbsp;

### Predict

This is for when you have trained a model and want to predict the emotion for an audio. Check out [model-backup branch](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/model-backup) or [release page](https://github.com/Renovamen/Speech-Emotion-Recognition/releases) for some pretrained models.


Argparse：

| Long option    | Option | Description                                                  |
| -------------- | ------ | ------------------------------------------------------------ |
| `--model_type` | `-mt`  | model type [ `svm` / `mlp` / `lstm` ] [ default is `svm` ]   |
| `--model_name` | `-mn`  | name of the model file that will be loaded [ default is `default` ] |
| `--feature`    | `-f`   | how to extract features [ `o`: Opensmile / `l`: librosa ] [ default is `o` ] |
| `--audio`      | `-a`   | path of the audio for predicting [ default is `default.wav` ] |

Example:

```python
python predict.py -mt 'svm' -mn 'SVM' -f 'o' -a 'test/happy.wav'
```

More examples can be found in [`example.sh`](example.sh).

&nbsp;

If you don't want to set parameters via command line:

```python
from misc.utils import load_model
from predict import predict

'''
input params:
	load_model_name: name of the model file you want to load
	model_name: model type (svm / mlp / lstm)
return:
	model: a loaded model
'''
model = load_model(load_model_name = "LSTM", model_name = "lstm")

'''
input params:
	model: a loaded model
	model_name: model type (svm / mlp / lstm)
	file_path: path of the audio you want to predict
	feature_method: how to extract features ('o': Opensmile / 'l': librosa)
return: 
	predicted results along with probabilities
'''
predict(model, model_name = "lstm", file_path = 'test/angry.wav', feature_method = 'l')
```



&nbsp;

### Functions

#### Radar Chart

Plot a radar chart for demonstrating predicted probabilities.

Source: [Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
from misc.utils import Radar
'''
Input:
    data_prob: probabilities
'''
Radar(result_prob)
```

&nbsp;

#### Play Audio

```python
from misc.utils import playAudio
playAudio(file_path)
```

&nbsp;

#### Plot Curve

Plot loss curve or accuracy curve.

```python
from misc.utils import plotCurve
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

#### Waveform

Plot a waveform for an audio file.

```python
from misc.utils import Waveform
Waveform(file_path)
```

&nbsp;

#### Spectrogram

Plot a spectrogram for an audio file.

```python
from Utils import Spectrogram
Spectrogram(file_path)
```

&nbsp;

## Other Contributors

- [@Zhaofan-Su](https://github.com/Zhaofan-Su)
- [@Guo Hui](https://github.com/guohui15661353950)