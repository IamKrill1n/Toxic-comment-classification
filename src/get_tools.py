from os import chdir, path, getcwd
import json
import re


if getcwd().endswith("src"):
    chdir(path.pardir)
if path.isfile("checkcwd"):
    print("Success")
else:
    raise Exception("Something went wrong. cwd=" + getcwd())


input_dir = 'kaggle\\input\\jigsaw-toxic-comment-classification-challenge\\'
tool_dir = 'text-preprocessing-tools-light\\'


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
    badwords_fix[fr' {v} '].append(r'([ ]{1,}' + re.escape(k) + r'[ ]{1,})')

for k in badwords_fix:
    badwords_fix[k] = re.compile(r'|'.join(badwords_fix[k]))
