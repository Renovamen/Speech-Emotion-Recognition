import argparse


def parse_prepro():

    paser = argparse.ArgumentParser(description = 'preprocessing options for speech emotion recognition')

    paser.add_argument(
        '-f', 
        '--feature', 
        type = str, 
        default = 'o',
        dest = 'feature', 
        help = "The method for features extracting: use 'o' to use opensmile or use 'l' to use librosa.")

    args = paser.parse_args()
    return args


def parse_train():

    paser = argparse.ArgumentParser(description = 'Speech Emotion Recognition')

    # svm / mlp / lstm
    paser.add_argument(
        '-mt', 
        '--model_type', 
        type = str, 
        default = 'svm',
        dest = 'model_type', 
        help = "The type of model (svm, mlp or lstm).")

    paser.add_argument(
        '-mn', 
        '--model_name', 
        type = str, 
        default = 'default',
        dest = 'model_name', 
        help = "The name of saved model file.")
    
    paser.add_argument(
        '-f', 
        '--feature', 
        type = str, 
        default = 'o',
        dest = 'feature', 
        help = "The method for features extracting: 'o' for opensmile, 'l' for librosa.")

    args = paser.parse_args()
    return args


def parse_pred():

    paser = argparse.ArgumentParser(description = 'Speech Emotion Recognition')

    # svm / mlp / lstm
    paser.add_argument(
        '-mt', 
        '--model_type', 
        type = str, 
        default = 'svm',
        dest = 'model_type', 
        help = "The type of model (svm, mlp or lstm).")

    paser.add_argument(
        '-mn', 
        '--model_name', 
        type = str, 
        default = 'default',
        dest = 'model_name', 
        help = "The name of saved model file.")
    
    paser.add_argument(
        '-f', 
        '--feature', 
        type = str, 
        default = 'o',
        dest = 'feature', 
        help = "The method for features extracting: 'o' for opensmile, 'l' for librosa.")

    paser.add_argument(
        '-a', 
        '--audio', 
        type = str, 
        default = 'default.wav',
        dest = 'audio', 
        help = "The path of audio which you want to predict.")

    args = paser.parse_args()
    return args