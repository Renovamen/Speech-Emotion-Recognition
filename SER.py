import numpy as np
from keras.utils import np_utils

from DNN_Model import LSTM_Model
from ML_Model import SVM_Model
from ML_Model import MLP_Model
from DNN_Model import CNN_Model

from Utilities import get_feature
from Utilities import get_data
from Utilities import load_model
from Utilities import Radar

DATA_PATH = 'DataSet/Berlin'
CLASS_LABELS = ("Angry", "Happy", "Neutral", "Sad")
# CLASS_LABELS = ("Angry", "Fearful", "Happy", "Neutral", "Sad", "Surprise")
# CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")

def LSTM():
    FLATTEN = False
    LOAD_MODEL = 'DNN'
    NUM_LABELS = len(CLASS_LABELS)

    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)

    print('-------------------------------- LSTM Start --------------------------------')
    model = LSTM_Model(input_shape = x_train[0].shape, num_classes = NUM_LABELS)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs = 1)
    model.evaluate(x_test, y_test)
    model.save_model("LSTM1")
    filename = '03-01-05-01-01-01-01.wav'


    result, result_prob = model.recognize_one(get_feature(filename))
    print('Recogntion: ', CLASS_LABELS[result])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)
    print('--------------------------------- LSTM End ---------------------------------')

    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    '''
    # 加载json
    model = load_model(model_name = "LSTM1", load_model = LOAD_MODEL)
    filename = '03-01-05-01-01-01-01.wav'

    result = np.argmax(model.predict(np.array([get_feature(filename)])))
    result_prob = model.predict(np.array([get_feature(filename)]))[0]
    print('Recogntion: ', CLASS_LABELS[result])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)
    '''

def CNN():
    FLATTEN = False
    LOAD_MODEL = "DNN"

    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    NUM_LABELS = len(CLASS_LABELS)
    
    print('-------------------------------- CNN Start --------------------------------')
    model = CNN_Model(input_shape = x_train[0].shape, num_classes = NUM_LABELS)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs = 1)
    model.evaluate(x_test, y_test)
    model.save_model("CNN1")

    print('-------------------------------- CNN Start --------------------------------')
    
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    '''
    # 加载json
    model = load_model(model_name = "CNN1", load_model = LOAD_MODEL)
    filename = '03-01-05-01-01-01-01.wav'

    test = np.array([get_feature(filename, flatten = FLATTEN)])
    in_shape = test[0].shape
    test = test.reshape(test.shape[0], in_shape[0], in_shape[1], 1)
    print('Recogntion: ', CLASS_LABELS[np.argmax(model.predict(test))])
    '''

def MLP():
    FLATTEN = True
    LOAD_MODEL = "MLP"
    NUM_LABELS = len(CLASS_LABELS)

    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN)
    model = MLP_Model() # 要用的方法（SVM / MLP）
    print('--------------------------------  Start --------------------------------')
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    # model.save_model("MLP1")
    filename = '03-01-05-01-01-01-01.wav'

    result, result_prob = model.recognize_one(get_feature(filename, flatten = FLATTEN))
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)

    print('---------------------------------- End ----------------------------------')
    
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    '''
    model = load_model(model_name = "MLP1", load_model = LOAD_MODEL)
    filename = '03-01-05-01-01-01-01.wav'
    result = model.predict(np.array([get_feature(filename, flatten = FLATTEN)]))
    result_prob = model.predict_proba(np.array([get_feature(filename, flatten = FLATTEN)]))
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)
    '''

