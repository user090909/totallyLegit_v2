from PIL import Image
import pytesseract
import os
import nltk
import re
from nltk.corpus import stopwords

def get_text(img):
    # image = Image.open(path)
    text = pytesseract.image_to_string(img, lang='rus')
    return text

def check_words(text: str, words: list, islower=True):
    if islower:
        text = text.lower()
    word_counter = 0
    for word in text.split():
        if word in words:
            word_counter += 1
    return word_counter


def normalize(string:str):
    snowball = nltk.SnowballStemmer(language="russian")
    stop_words = set(stopwords.words('russian'))

    lower_string = string.lower()
    no_number_string = re.sub(r'\d+', '', lower_string)
    no_punc_string = re.sub(r'[^\w\s]', '', no_number_string)
    no_wspace_string = no_punc_string.strip()
    lst_string = [no_wspace_string][0].split()
    no_stpwords_string = ""
    for i in lst_string:
        if i not in stop_words:
            no_stpwords_string += i + ' '

    no_stpwords_string = no_stpwords_string[:-1]

    l = ""
    for i in no_stpwords_string.split(" "):
        l += snowball.stem(i) + " "

    return l[:-1]