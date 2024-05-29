import numpy as np
import pandas as pd
from joblib import load
from scipy import sparse

import sys
sys.path.append('src')

from os import chdir, path, getcwd
for i in range(10):
    if path.isfile("checkcwd"):
        break
    chdir(path.pardir)
if path.isfile("checkcwd"):
    pass
else:
    raise Exception("Something went wrong. cwd=" + getcwd())

from data_preprocessing.get_tools import clean_text_light

class LogisticRegression:
    def __init__(self) -> None:
        self.char_vectorizer = load('model_checkpoint/logistic_regression/char_vectorizer.bin')
        self.word_vectorizer = load('model_checkpoint/logistic_regression/word_vectorizer.bin')
        self.loaded_models = []
        for i in range(6):
            self.loaded_models.append(load('model_checkpoint/logistic_regression/logistic_regression_%d.bin' % i))
    
    def predict(self, query: str) -> 'np.ndarray':
        
        query = clean_text_light(query)

        if query == '':
            result = np.array([0, 0, 0, 0, 0, 0])
            return result
        
        X_test = pd.Series([query])

        test_word_features = self.word_vectorizer.transform(X_test)
        test_char_features = self.char_vectorizer.transform(X_test)

        X_te = sparse.hstack([test_word_features, test_char_features])

        predict = np.zeros((1, 6))

        for (i, model) in enumerate(self.loaded_models):
            predict[:, i] = model.predict_proba(X_te)[:, 1]

        return predict.reshape(-1, )
    

def main() -> None:
    demo = LogisticRegression()
    while True:
        query = str(input('>>> '))
        res = demo.predict(query)
        print(res)

if __name__ == '__main__':
    main()