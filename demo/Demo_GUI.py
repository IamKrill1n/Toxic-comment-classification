import gradio as gr
import pandas as pd 
import plotly.express as px 
from HybridRNN import HybridRNN
from LogisticRegression import LogisticRegression
from NaiveBayes import NaiveBayes
from NBLR import NBLR
from NBSVM import NBSVM
from RNN import RNN 
from SVM import SVM


# Function to load models
print('Loading models...')
hrnn = HybridRNN()
lore = LogisticRegression()
nb = NaiveBayes()
nblr = NBLR()
nbsvm = NBSVM()
rnn = RNN()
svm = SVM()
print('Models loaded.')

def create_dataframe(classes, probabilities):
    dataframe =  pd.DataFrame(
        {
            "Classes": classes,
            "Toxicity probabilities": probabilities
            
        }
    )
    
    dataframe['Color'] = ['red' if proba >= 50 else 'blue' for proba in probabilities]
    return dataframe 
    
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# Define the predict function
def toxicity_proba(comment, model):
    if model == "Hybrid RNN":
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in hrnn.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == "Logistic Regression":
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in lore.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == "Naive Bayes":
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in nb.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == "NBSVM":
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in nbsvm.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == 'NBLR':
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in nblr.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == 'RNN':
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in rnn.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})
    
    elif model == "SVM":
        dataframe = create_dataframe(classes, [float("{:.4f}".format(proba))*100 for proba in svm.predict(comment).tolist()])
        return px.bar(dataframe, x='Classes', y='Toxicity probabilities', color='Color', color_discrete_map={'red': 'red', 'blue': 'blue'})


# Define the Gradio interface
demo = gr.Interface(
    fn=toxicity_proba,
    inputs=[
        gr.Textbox(label='Comment', placeholder="Enter comment here"),
        gr.Radio(["Hybrid RNN", "Logistic Regression", "Naive Bayes", "NBSVM", "NBLR", "RNN", "SVM"], label="Model")
    ],
    outputs=gr.Plot()
)

# Launch the interface
demo.launch()
