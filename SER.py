import numpy as np
from keras.utils import np_utils

from DNN_Model import LSTM_Model
from ML_Model import SVM_Model
from ML_Model import MLP_Model
from DNN_Model import CNN_Model

from Utilities import get_feature
from Utilities import get_feature_svm
from Utilities import get_data
from Utilities import load_model
from Utilities import Radar

DATA_PATH = 'DataSet/CASIA'
# CLASS_LABELS = ("Angry", "Happy", "Neutral", "Sad")
# CLASS_LABELS = ("Angry", "Fearful", "Happy", "Neutral", "Sad", "Surprise")
CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")

def LSTM(file_path: str):
    FLATTEN = False
    LOAD_MODEL = 'DNN'
    NUM_LABELS = len(CLASS_LABELS)
    SVM = False
    '''
    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN, _svm = SVM)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)

    print('-------------------------------- LSTM Start --------------------------------')
    model = LSTM_Model(input_shape = x_train[0].shape, num_classes = NUM_LABELS)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs = 100)
    model.evaluate(x_test, y_test)
    model.save_model("LSTM1")

    result, result_prob = model.recognize_one(get_feature(file_path))
    print('Recogntion: ', CLASS_LABELS[result])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)
    print('--------------------------------- LSTM End ---------------------------------')
    '''
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    
    # 加载json
    model = load_model(model_name = "LSTM1", load_model = LOAD_MODEL)

    result = np.argmax(model.predict(np.array([get_feature(file_path)])))
    result_prob = model.predict(np.array([get_feature(file_path)]))[0]
    print('Recogntion: ', CLASS_LABELS[result])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)


def CNN(file_path: str):
    FLATTEN = False
    LOAD_MODEL = "DNN"
    NUM_LABELS = len(CLASS_LABELS)
    SVM = False
    '''
    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN, _svm = SVM)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    
    print('-------------------------------- CNN Start --------------------------------')
    model = CNN_Model(input_shape = x_train[0].shape, num_classes = NUM_LABELS)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs = 50)
    model.evaluate(x_test, y_test)
    model.save_model("CNN1")

    print('-------------------------------- CNN Start --------------------------------')
    '''
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    
    # 加载json
    model = load_model(model_name = "CNN1", load_model = LOAD_MODEL)

    test = np.array([get_feature(file_path, flatten = FLATTEN)])
    in_shape = test[0].shape
    test = test.reshape(test.shape[0], in_shape[0], in_shape[1], 1)
    result = np.argmax(model.predict(test))
    result_prob = model.predict(test)[0]
    print('Recogntion: ', CLASS_LABELS[result])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)


def MLP(file_path: str):
    FLATTEN = True
    LOAD_MODEL = "ML"
    NUM_LABELS = len(CLASS_LABELS)
    SVM = False
    '''
    x_train, x_test, y_train, y_test = get_data(DATA_PATH, class_labels = CLASS_LABELS, flatten = FLATTEN, _svm = SVM)
    model = MLP_Model()
    print('--------------------------------  Start --------------------------------')
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model("MLP1")

    result, result_prob = model.recognize_one(get_feature(file_path, flatten = FLATTEN))
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)

    print('---------------------------------- End ----------------------------------')
    '''
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    model = load_model(model_name = "MLP1", load_model = LOAD_MODEL)

    result = model.predict(np.array([get_feature(file_path, flatten = FLATTEN)]))
    result_prob = model.predict_proba(np.array([get_feature(file_path, flatten = FLATTEN)]))[0]
    print('Recogntion: ', CLASS_LABELS[result[0]])
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)

def SVM(file_path: str):
    
    FLATTEN = True
    LOAD_MODEL = "ML"
    NUM_LABELS = len(CLASS_LABELS)
    SVM = True
    '''
    x_train, x_test, y_train, y_test = get_data(DATA_PATH, mfcc_len = 48, class_labels = CLASS_LABELS, flatten = FLATTEN, _svm = SVM)
    model = SVM_Model()
    print('--------------------------------  Start --------------------------------')
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.save_model("SVM1")

    result, result_prob = model.recognize_one(get_feature_svm(file_path, mfcc_len = 48))
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)
 
    print('---------------------------------- End ----------------------------------')
    '''
    '''
    ---------------------------- 加载模型 ----------------------------
    '''
    
    model = load_model(model_name = "SVM1", load_model = LOAD_MODEL)

    result = model.predict(np.array([get_feature_svm(file_path, mfcc_len = 48)]))
    result_prob = model.predict_proba(np.array([get_feature_svm(file_path, mfcc_len = 48)]))[0]
    print('Recogntion: ', result)
    print('Probability: ', result_prob)
    Radar(result_prob, CLASS_LABELS, NUM_LABELS)

SVM("03a04Wc.wav")