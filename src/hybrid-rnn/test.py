import keras
import numpy as np
import re

from joblib import load

from os import chdir, path, getcwd
if getcwd().endswith("src"):
    chdir(path.pardir)
if path.isfile("checkcwd"):
    print("Success")
else:
    raise Exception("Something went wrong. cwd=" + getcwd())




loaded_model = keras.saving.load_model('src/hybrid-rnn/hybrid.keras')

ss = load('src/hybrid-rnn/scaler.bin')
tokenizer = load('src/hybrid-rnn/tokenizer.bin')

comment_text = str(input('>>> ')).strip()

if comment_text == '':
    print('empty comment')
    exit()

total_length = len(comment_text)
capitals = sum(1 for c in comment_text if c.isupper())
caps_vs_length = capitals / total_length

num_words = len(re.findall(r'\S+', comment_text))
print(num_words)

num_unique_words = len(set(w for w in comment_text.split()))
words_vs_unique = num_unique_words / num_words

test_features = np.array([caps_vs_length, words_vs_unique]).reshape(1, -1)
test_features = ss.transform(test_features)