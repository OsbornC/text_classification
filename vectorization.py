import pyprind
import pandas as pd
import os
import numpy as np
 
def textExtraction():
    pbar = pyprind.ProgBar(50000)
    labels = {"pos":1,"neg":0}
    data = pd.DataFrame()
    for s in ("test","train"):
        for l in ("pos","neg"):
            path = "./%s/%s"%(s,l)
            for file in os.listdir(path):
                with open(os.path.join(path,file),"r",encoding="utf-8") as f:
                    txt = f.read()
                data = data.append([[txt,labels[l]]],ignore_index=True)
                pbar.update()
    data.columns = ["review","sentiment"]
    np.random.seed(0)
    data = data.reindex(np.random.permutation(data.index))
    data.to_csv("./movie_data.csv",index=False)
textExtraction()

movie_data = pd.read_csv('movie_data.csv')

movie_data.head()

texts = movie_data['review'].copy()

import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
def text_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    
    return " ".join(text)

texts = texts.apply(text_process)

vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(texts)

movie_data['vectors'] = features.toarray()

from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words='english')
X = vec.fit_transform(texts)
X.toarray()

X.get_feature_names()

