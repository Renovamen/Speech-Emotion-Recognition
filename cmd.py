import argparse

from SER import Train, Predict
from Utils import load_model

def cmd():

    paser = argparse.ArgumentParser(description = 'Speech Emotion Recognition')

    paser.add_argument(
        '-o',
        '--option',
        type = str,
        dest = 'option',
        help = "Use 'p' to predict directly or use 't' to train a model.")

    paser.add_argument(
        '-mt', 
        '--model_type', 
        type = str, 
        dest = 'model_type', 
        help = "The type of model.")

    paser.add_argument(
        '-mn', 
        '--model_name', 
        type = str, 
        dest = 'model_name', 
        help = "The name of saved model file.")
    
    paser.add_argument(
        '-l', 
        '--load', 
        type = bool, 
        dest = 'load', 
        help = "Whether to load exist features.")
    
    paser.add_argument(
        '-f', 
        '--feature', 
        type = str, 
        dest = 'feature', 
        help = "The method for features extracting: use 'o' to use opensmile or use 'l' to use librosa.")

    paser.add_argument(
        '-a', 
        '--audio', 
        type = str, 
        dest = 'audio', 
        help = "The path of audio which you want to predict.")


    args = paser.parse_args()

    option = args.option.lower() # p / t
    model_type = args.model_type if args.model_type else 'svm' # svm / mlp / lstm
    model_name = args.model_name if args.model_name else 'default' 
    load = args.load if args.load else True # True / False
    feature = args.feature if args.feature else 'o' # o / l
    audio = args.audio if args.audio else 'default.wav'

    # 预测
    if option == 'p':
        model = load_model(load_model_name = model_name, model_name = model_type)
        Predict(model, model_name = model_type, file_path = audio, feature_method = feature)
    
    # 训练
    elif option == 't':
        Train(model_name = model_type, save_model_name = model_name, if_load = load, feature_method = feature)

    else:
        print("Wrong option. 'p' for predicting, 't' for training")
        return


if __name__ == '__main__':
    cmd()