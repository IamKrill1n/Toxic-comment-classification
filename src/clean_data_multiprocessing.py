import re
import multiprocessing as mp
import pandas as pd
import string

from src.get_tools import lemmatizer, lengthening_pattern, badwords_fix, APPO, stopwords


def remove_special_characters(text: str) -> str:

    # Remove non ascii characters: this will remove any emoji too
    text = re.sub(r'[^\x00-\x7f]', r' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove emojis
    # text = re.sub(r'[\U00010000-\U0010ffff]', r'', text, flags=re.UNICODE)

    # Remove link
    text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', r' ', text)

    # Remove leaky elements like ip, user
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', r' ', text)

    # Remove \n\r
    text = re.sub(r'/\\n+|\\r+|\n+|\r+/', r' ', text)

    # Removing usernames
    text = re.sub(r'\[\[.*\]', r' ', text)

    # Reduce lengthening
    # https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    text = re.sub(lengthening_pattern, r'\1\1', text)

    # Check for common bad words 'workaround'
    text = f' {text} '
    for k, v in badwords_fix.items():
        text = re.sub(v, k, text)
    
    # Remove punctuation
    text = re.sub(r'[]!"$%&()*+,./:;=#@?[\\^_`{|}~-]+', r' ', text)

    # Replace appos
    words=[APPO[word] if word in APPO else word for word in text.split()]
    text = ' '.join(words)
    text = re.sub(r"'", '', text)

    # Remove all special characters
    # text = re.sub(r"[^\w']", ' ', text)

    return text

def split_numbers_from_characters(text):
    # Split numbers and characters
    parts = re.split('(\d+)', text)
    
    # If there was a number and a character part, add a space between them
    if len(parts) > 1:
        return ' '.join(parts)
    
    # If there was no number, return the original text
    return text

def remove_numbers(text):
    remove_digits = str.maketrans('', '', string.digits)
    return text.translate(remove_digits)

def clean_text_vanila(text):
    # Remove special characters
    text = remove_special_characters(text)
    text = remove_numbers(text)
    return text

def clean_text_light(text):
    # Remove special characters
    text = remove_special_characters(text)
    text = remove_numbers(text)
    # lemmatization
    words = [lemmatizer.get(word, word) for word in text.split()]
    # Remove numbers again as lemma might have numbers
    words = remove_numbers(' '.join(words)).split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords]
    return ' '.join(words)


def clean_data_multiprocessing(df: 'pd.DataFrame') -> None:
    with mp.Pool(mp.cpu_count()) as pool:
        df['cleaned_data'] = pool.map(clean_text_light, df['comment_text'])