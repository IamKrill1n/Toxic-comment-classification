import gradio as gr
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

# Define the predict function
def toxicity_proba(comment, model):
    if model == "Hybrid RNN":
        return hrnn.predict(comment)
    elif model == "Logistic Regression":
        return lore.predict(comment)
    elif model == "Naive Bayes":
        return nb.predict(comment)
    elif model == "NBSVM":
        return nbsvm.predict(comment)
    elif model == 'NBLR':
        return nblr.predict(comment)
    elif model == 'RNN':
        return rnn.predict(comment)
    elif model == "SVM":
        return svm.predict(comment)

# Define the Gradio interface
demo = gr.Interface(
    fn=toxicity_proba,
    inputs=[
        gr.Textbox(label='Comment', placeholder="Enter comment here"),
        gr.Radio(["Hybrid RNN", "Logistic Regression", "Naive Bayes", "NBSVM", "NBLR", "RNN", "SVM"], label="Model")
    ],
    outputs=gr.Textbox(label="Result")
)

# Launch the interface
demo.launch()
