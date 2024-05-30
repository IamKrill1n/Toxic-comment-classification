import gradio as gr
from joblib import load
import numpy as np 
import sys 
sys.path.append('src')

from data_preprocessing.get_tools import clean_text_light
class TFIDF: 
    def __init__(self) -> None:
        self.loaded_models = []
        for i in range(6):
            self.loaded_models.append(load(f'src/NaiveBayes/nb_models/naive_bayes_tfidf/naive_bayes_tfidf{i}.joblib'))
    
    def predict(self, query: str) -> 'np.ndarray':
        
        query = clean_text_light(query)
        
        if query == '':
            result = np.array([0, 0, 0, 0, 0, 0])
            return result 
        
        X_test = [query]
        predict = np.zeros((1, 6))
        
        for (i, model) in enumerate(self.loaded_models):
            predict[:, i] = model.predict_proba(X_test)[:, -1]
        
        return predict.reshape(-1, )
    
class BoW: 
    def __init__(self) -> None:
        self.loaded_models = []
        for i in range(6):
            self.loaded_models.append(load(f'src/NaiveBayes/nb_models/naive_bayes_bow/naive_bayes_bow{i}.joblib'))
    
    def predict(self, query: str) -> 'np.ndarray':
        
        query = clean_text_light(query)
        
        if query == '':
            result = np.array([0, 0, 0, 0, 0, 0])
            return result 
        
        X_test = [query]
        predict = np.zeros((1, 6))
        
        for (i, model) in enumerate(self.loaded_models):
            predict[:, i] = model.predict_proba(X_test)[:, -1]
        
        return predict.reshape(-1, )
    
class BinBoW: 
    def __init__(self) -> None:
        self.loaded_models = []
        for i in range(6):
            self.loaded_models.append(load(f'src/NaiveBayes/nb_models/naive_bayes_binary_bow/naive_bayes_binary_bow{i}.joblib'))
    
    def predict(self, query: str) -> 'np.ndarray':
        
        query = clean_text_light(query)
        
        if query == '':
            result = np.array([0, 0, 0, 0, 0, 0])
            return result 
        
        X_test = [query]
        predict = np.zeros((1, 6))
        
        for (i, model) in enumerate(self.loaded_models):
            predict[:, i] = model.predict_proba(X_test)[:, -1]
        
        return predict.reshape(-1, )
    
# Function to load models

print('Loading models...')
tfidf_model = TFIDF()
bow_model = BoW()
binary_bow_model = BinBoW()
print('Models loaded.')

# Define the predict function
def predict(comment, model):
    if model == "TF-IDF":
        return tfidf_model.predict(comment)
    elif model == "BoW":
        return bow_model.predict(comment)
    elif model == "Binary BoW":
        return binary_bow_model.predict(comment)

# Define the Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label='Comment', placeholder="Enter comment here"),
        gr.Radio(["TF-IDF", "BoW", "Binary BoW"], label="Model")
    ],
    outputs=gr.Textbox(label="Result")
)

# Launch the interface
demo.launch()
