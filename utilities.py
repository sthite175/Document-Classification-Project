import pandas as pd
import numpy as np
import json
import pickle
import config

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from unidecode import unidecode
from contractions import fix
from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import re
from PIL import Image
import pytesseract



# stopwords
stopwords_list = stopwords.words("english")

def preprocess_data(text):
    text = text.lower()
    text = text.replace("\n"," ").replace("\t"," ")
    text = re.sub("\s+"," ",text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokens
    tokens = word_tokenize(text)
    
    data = [i for i in tokens if i not in punctuation]
    data = [i for i in data if i not in stopwords_list]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for i in data:
        word = lemmatizer.lemmatize(i)
        final_text.append(word)
        
    return " ".join(final_text)


def image_load_and_preprocess(path):
    image = Image.open(path)
    text = pytesseract.image_to_string(image)

    text_data = preprocess_data(text)

    return text_data



#print(preprocess_data("\n\nThe Contribution of Tobacco\nConstituents to Phenol Yield |..*\nof Cigarettes’\n\n \n\n4\n\nJ. H. Bell, A. ©. Saunders and A. W. Spears\n\n \n\nResearch Division, P. Lorillard Company, Inc.\nGreensboro, North Carolina, U.S."))



