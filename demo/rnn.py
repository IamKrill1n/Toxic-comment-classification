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
    def __init__(self, model_path, tokenizer_path = 'model_checkpoint/tokenizer.json'):
        self.model = load_model(model_path)
        with open(tokenizer_path) as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)
    
    def predict(self, comments):
        comments = [clean_text_vanilla(comment) for comment in comments]
        sequences = self.tokenizer.texts_to_sequences(comments)
        X = pad_sequences(sequences, maxlen=100)
        return self.model.predict([X])

if __name__ == "__main__":
    model = RNN('model_checkpoint/rnn.keras')
    print(model.predict(['I fucking love this movie']))