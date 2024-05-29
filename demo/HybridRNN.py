import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

import sys
sys.path.append('src')

import tensorflow as tf


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



class HybridRNN:

    def __init__(self, use_gpu: bool = False) -> None:

        if use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
                    print('Success. Using GPU...')
                except RuntimeError as e:
                    print(e)
            else:
                print('No GPU detected. Using CPU...')
        else:
            tf.config.set_visible_devices([], 'GPU')

        self.loaded_model = keras.saving.load_model('model_checkpoint/hybrid_rnn/hybrid.keras')
        self.ss = load('model_checkpoint/hybrid_rnn/scaler.bin')
        self.tokenizer = load('model_checkpoint/hybrid_rnn/tokenizer.bin')


    def predict(self, query: str) -> 'np.ndarray':

        query = clean_text_light(query)

        if query == '':
            result = np.array([0, 0, 0, 0, 0, 0])
            return result

        list_sentences_test = [query]
        list_tokenized_test = self.tokenizer.texts_to_sequences(list_sentences_test)
        X_te = pad_sequences(list_tokenized_test, maxlen=constants.MAXLEN)

        total_length = len(query)
        capitals = sum(1 for c in query if c.isupper())
        caps_vs_length = capitals / total_length

        num_words = len(re.findall(r'\S+', query))

        num_unique_words = len(set(w for w in query.split()))
        words_vs_unique = num_unique_words / num_words

        test_features = np.array([caps_vs_length, words_vs_unique]).reshape(1, -1)
        test_features = self.ss.transform(test_features)

        predict = self.loaded_model.predict([X_te, test_features])
        return predict.reshape(-1, )
    

def main() -> None:
    demo = HybridRNN(use_gpu=False)
    while True:
        query = str(input('>>> '))
        res = demo.predict(query)
        print(res)

if __name__ == '__main__':
    main()