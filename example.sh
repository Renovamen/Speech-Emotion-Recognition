# preprocess
# using Opensmile
python preprocess.py -f 'o'
# using Librosa
python preprocess.py -f 'l'

# train
# on features extracted by Opensmile
python train.py -mt 'svm' -mn 'SVM_OPENSMILE' -f 'o'
python train.py -mt 'mlp' -mn 'MLP_OPENSMILE' -f 'o'
python train.py -mt 'lstm' -mn 'LSTM_OPENSMILE' -f 'o'
# on features extracted by Librosa
python train.py -mt 'svm' -mn 'SVM_LIBROSA' -f 'l'
python train.py -mt 'mlp' -mn 'MLP_LIBROSA' -f 'l'
python train.py -mt 'lstm' -mn 'LSTM_LIBROSA' -f 'l'

# predict
# using features extracted by Opensmile
python predict.py -mt 'svm' -mn 'SVM_OPENSMILE' -f 'o' -a 'test/happy.wav'
python predict.py -mt 'mlp' -mn 'MLP_OPENSMILE' -f 'o' -a 'test/neutral.wav'
python predict.py -mt 'lstm' -mn 'LSTM_OPENSMILE' -f 'o' -a 'test/angry.wav'
# using features extracted by Librosa
python predict.py -mt 'svm' -mn 'SVM_LIBROSA' -f 'l' -a 'test/surprise.wav'
python predict.py -mt 'mlp' -mn 'MLP_LIBROSA' -f 'l' -a 'test/sad.wav'
python predict.py -mt 'lstm' -mn 'LSTM_LIBROSA' -f 'l' -a 'test/fear.wav'