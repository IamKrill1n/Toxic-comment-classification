import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np, pandas as pd
import json
import sys
sys.path.append('src')

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from data_preprocessing.get_tools import clean_text_vanilla

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class RNN:
    def __init__(self, model_path = 'model_checkpoint/rnn/glove300_lstm.keras', tokenizer_path = 'model_checkpoint/rnn/tokenizer.json'):
        self.model = load_model(model_path)
        with open(tokenizer_path) as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)
    
    def predict(self, comment) -> 'np.ndarray':
        comment = clean_text_vanilla(comment)
        sequences = self.tokenizer.texts_to_sequences([comment])
        X = pad_sequences(sequences, maxlen=100)
        return self.model.predict([X]).reshape(-1, )

if __name__ == "__main__":
    model = RNN('model_checkpoint/rnn/glove300_lstm.keras')
    while True:
        query = str(input('>>> '))
        res = model.predict(query)
        print(res)