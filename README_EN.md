# Speech Emotion Recognition 

Speech emotion recognition using LSTM, CNN, SVM and MLP, implemented in Keras.

We have improved the feature extracting method and achieved higher accuracy (about 80%). The original version is backed up under [First-Version](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/First-Version) branch.

English Document | [中文文档](README.md)


&nbsp;

## Environments

- Python 3.8
- Keras & TensorFlow 2


&nbsp;

## Structure

```
├── models/                // models
│   ├── common.py          // base class for all models
│   ├── dnn                // neural networks
│   │   ├── dnn.py         // base class for all neural networks models
│   │   ├── cnn.py         // CNN
│   │   └── lstm.py        // LSTM
│   └── ml.py              // SVM & MLP
├── extract_feats/         // features extraction
│   ├── librosa.py         // extract features using librosa
│   └── opensmile.py       // extract features using Opensmile
├── utils/
│   ├── files.py           // setup dataset (classify and rename)
│   ├── opts.py            // argparse
│   └── plot.py            // plot graphs
├── features/              // store extracted features
├── config.py              // configure parameters
├── train.py               // train
├── predict.py             // recognize the emotion of a given audio
└── preprocess.py          // data preprocessing (extract features and store them locally)
```


&nbsp;

## Requirments

### Python

- [TensorFlow 2](https://github.com/tensorflow/tensorflow) / [Keras](https://github.com/keras-team/keras): LSTM & CNN (`tensorflow.keras`)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): SVM & MLP, split data into training set and testing set
- [joblib](https://github.com/joblib/joblib)：save and load models trained by scikit-learn
- [librosa](https://github.com/librosa/librosa): extract features, waveform
- [SciPy](https://github.com/scipy/scipy): spectrogram
- [pandas](https://github.com/pandas-dev/pandas): load features
- [Matplotlib](https://github.com/matplotlib/matplotlib): plot graphs
- [NumPy](https://github.com/numpy/numpy)

### Tools

- [Opensmile](https://github.com/naxingyu/opensmile): extract features


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

Parameters can be configured in the config files (YAML) under [`configs/`](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/master/configs).

It should be noted that, currently only the following 6 Opensmile standard feature sets are supported:

- `IS09_emotion`: [The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), 384 features;
- `IS10_paraling`: [The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf), 1582 features;
- `IS11_speaker_state`: [The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf), 4368 features;
- `IS12_speaker_trait`: [The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf), 6125 features;
- `IS13_ComParE`: [The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf), 6373 features;
- `ComParE_2016`: [The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), 6373 features.

You may should modify item `FEATURE_NUM` in [`extract_feats/opensmile.py`](extract_feats/opensmile.py) if you want to use other feature sets.

&nbsp;

### Preprocess

First of all, you should extract features of each audio in dataset and store them locally. Features extracted by Opensmile will be saved in `.csv` files and by librosa will be saved in `.p` files.

```python
python preprocess.py --config configs/example.yaml
```

where `configs/test.yaml` is the path to your config file

&nbsp;

### Train

The path of the datasets can be configured in [`config.py`](config.py). Audios which express the same emotion should be put in the same folder (you may want to refer to [`utils/files.py`](utils/files.py) when setting up datasets), for example:

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

Then:

```python
python train.py --config configs/example.yaml
```

&nbsp;

### Predict

This is for when you have trained a model and want to predict the emotion for an audio. Check out [checkpoints branch](https://github.com/Renovamen/Speech-Emotion-Recognition/tree/checkpoints) or [release page](https://github.com/Renovamen/Speech-Emotion-Recognition/releases) for some checkpoints.

First modify following things in [`predict.py`](predict.py):

```python
audio_path = 'str: path_to_your_audio'
```

Then:

```python
python predict.py --config configs/example.yaml
```


&nbsp;

### Functions

#### Radar Chart

Plot a radar chart for demonstrating predicted probabilities.

Source: [Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
import utils

"""
Args:
    data_prob (np.ndarray): probabilities
    class_labels (list): labels
"""
utils.radar(data_prob, class_labels)
```

&nbsp;

#### Play Audio

```python
import utils

utils.play_audio(file_path)
```

&nbsp;

#### Plot Curve

Plot loss curve or accuracy curve.

```python
import utils

"""
Args:
    train (list): loss or accuracy on train set
    val (list): loss or accuracy on validation set
    title (str): title of figure
    y_label (str): label of y axis
"""
utils.curve(train, val, title, y_label)
```

&nbsp;

#### Waveform

Plot a waveform for an audio file.

```python
import utils

utils.waveform(file_path)
```

&nbsp;

#### Spectrogram

Plot a spectrogram for an audio file.

```python
import utils

utils.spectrogram(file_path)
```


&nbsp;

## Other Contributors

- [@Zhaofan-Su](https://github.com/Zhaofan-Su)
- [@Guo Hui](https://github.com/guohui15661353950)
