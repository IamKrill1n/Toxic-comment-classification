import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append('src')

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

from joblib import load

from os import chdir, path, getcwd
for i in range(10):
    if path.isfile("checkcwd"):
        break
    chdir(path.pardir)
if path.isfile("checkcwd"):
    pass
else:
    raise Exception("Something went wrong. cwd=" + getcwd())

import constants
from data_preprocessing.get_tools import clean_text_light




loaded_model = keras.saving.load_model('model_checkpoint/hybrid_rnn/hybrid.keras')

ss = load('model_checkpoint/hybrid_rnn/scaler.bin')
tokenizer = load('model_checkpoint/hybrid_rnn/tokenizer.bin')

while True:
    comment_text = str(input('>>> '))

    comment_text = clean_text_light(comment_text)

    if comment_text == '':
        # print('empty comment')
        result = np.array([0, 0, 0, 0, 0, 0])
        print(result)
        continue



    list_sentences_test = [comment_text]
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=constants.MAXLEN)



    total_length = len(comment_text)
    capitals = sum(1 for c in comment_text if c.isupper())
    caps_vs_length = capitals / total_length

    num_words = len(re.findall(r'\S+', comment_text))

    num_unique_words = len(set(w for w in comment_text.split()))
    words_vs_unique = num_unique_words / num_words

    test_features = np.array([caps_vs_length, words_vs_unique]).reshape(1, -1)
    test_features = ss.transform(test_features)



    predict = loaded_model.predict([X_te, test_features])
    print(predict.reshape(-1, ))
    # print(type(predict))