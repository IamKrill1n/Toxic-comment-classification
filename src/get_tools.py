from os import chdir, path, getcwd
import json
import re
import string


if getcwd().endswith("src"):
    chdir(path.pardir)
if path.isfile("checkcwd"):
    print("Success")
else:
    raise Exception("Something went wrong. cwd=" + getcwd())


input_dir = 'kaggle/input/jigsaw-toxic-comment-classification-challenge/'
tool_dir = 'text-preprocessing-tools-light/'


with open(tool_dir + 'lemmatization-en.json') as file:
    lemmatizer = json.load(file)

with open(tool_dir + 'stopwords.txt') as file:
    stopwords = file.read().splitlines()

# adapted from https://code.google.com/archive/p/badwordslist/downloads
with open(tool_dir + "badwords_data.json") as file:
    badwords_data = json.load(file)

with open(tool_dir + 'appo.json') as file:
    APPO = json.load(file)


lengthening_pattern = re.compile(r'(.)\1{2,}')


from collections import defaultdict
badwords_fix = defaultdict(list)
for k, v in badwords_data.items():
    # wildcard handling
    if k.endswith(r'*'):
        if k.startswith(r'*'):
            badwords_fix[fr' {v} '].append(r'( [\x21-\x7e]{0,}' + re.escape(k.strip('*')) + r'[\x21-\x7e]{0,} )')
        else:
            badwords_fix[fr' {v} '].append(r'( ' + re.escape(k.strip('*')) + r'[\x21-\x7e]{0,} )')
    else:
        badwords_fix[fr' {v} '].append(r'( ' + re.escape(k) + r' )')

for k in badwords_fix:
    badwords_fix[k] = re.compile(r'|'.join(badwords_fix[k]))


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
    text = re.sub(r'//n+|/r+|\n+|\r+/', r' ', text)

    # Removing usernames
    text = re.sub(r'\[\[.*\]', r' ', text)

    # Reduce lengthening
    # https://rustyonrampage.github.io/text-mining/2017/11/28/spelling-correction-with-python-and-nltk.html
    text = re.sub(lengthening_pattern, r'\1\1', text)

    # Check for common bad words 'workaround'
    text = re.sub(r' ', r'   ', text)
    text = f' {text} '
    for k, v in badwords_fix.items():
        text = re.sub(v, k, text)
    
    # Remove punctuation
    text = re.sub(r'[]!"$%&()*+,./:;=#@?[/^_`{|}~-]+', r' ', text)

    # Replace appos
    words=[APPO[word] if word in APPO else word for word in text.split()]
    text = ' '.join(words)
    text = re.sub(r"'", '', text)

    # Remove all special characters
    # text = re.sub(r"[^\w']", ' ', text)

    return text

def split_numbers_from_characters(text: str) -> str:
    # Split numbers and characters
    parts = re.split('(\d+)', text)
    
    # If there was a number and a character part, add a space between them
    if len(parts) > 1:
        return ' '.join(parts)
    
    # If there was no number, return the original text
    return text

def remove_numbers(text: str) -> str:
    remove_digits = str.maketrans('', '', string.digits)
    return text.translate(remove_digits)

def clean_text_vanila(text: str) -> str:
    # Remove special characters
    text = remove_special_characters(text)
    text = remove_numbers(text)
    return text

def clean_text_light(text: str) -> str:
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


def main() -> None:
    print(clean_text_vanila('dumb fuck say non sence hey dumb fuck know retard look like go look mirror fuck iamgodfuckyouassbiteneverdaremesswithkingvodious fuck youcantstopmeyoufuckingretardiwishyouwereheresoi couldstickmyfootupyourassanduseyouasanoddsandal'))

if __name__ == '__main__':
    main()