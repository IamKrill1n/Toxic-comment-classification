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

with open('model_checkpoint/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    
def query(comments, model, tokenizer):
    comments = [clean_text_vanilla(comment) for comment in comments]
    sequences = tokenizer.texts_to_sequences(comments)
    X = pad_sequences(sequences, maxlen=100)
    return model.predict([X])

if __name__ == "__main__":
    model = load_model('model_checkpoint/glove_lstm.keras')
    comments = input().split('\n')
    print(comments)
    print(query(comments, model, tokenizer))