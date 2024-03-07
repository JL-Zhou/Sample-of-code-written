import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import PorterStemmer

warnings.filterwarnings(action='ignore')



#from spacy import displacy


#NER = spacy.load("en_core_web_sm")

df = pd.read_csv('class1.csv')
df1 = pd.read_csv('class2.csv')
df2 = pd.read_csv('class3.csv')

#data = []

#for i in sent_tokenize(df['Description'][0]):
 #   temp = []

    # tokenize the sentence into words
  #  for j in word_tokenize(i):
   #     temp.append(j.lower())

    #data.append(temp)

#def sent_to_words(sentences):
   # for sentence in sentences:
    #    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

#text_data = sent_to_words(df['Description'])


#w2v_model = gensim.models.Word2Vec(text_data, vector_size=100, min_count=1, window=5, iter=100)
#df["Description"].apply(lambda text: np.mean([w2v_model.wv[word] for word in text.split() if word in w2v_model.wv]))
#print(df.Description)

#res = df['Description'][0].split()
km =[]
for i in range(len(df2)):
    k =remove_stopwords(df2['Description'][i])
    km.append(k)




ps = PorterStemmer()

sentence = "Programmers program with programming languages"
#words = word_tokenize(k)

#print(words)

#for w in words:
 #   print(w, " : ", ps.stem(w))
#k = []
#for i in range(len(df)):
 #   res = df['Description'][i].split()
  #  k.append = res
#print(k)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
d =[]
for i in range(len(km)):
    vectors = vectorizer.fit_transform([km[i]])
    v = np.mean(vectors)
    d.append(v)

df2['d_score']=d
df2.to_csv('file_name2.csv')





#print(np.mean(vectors))


# Create CBOW model
#model1 = gensim.models.Word2Vec(data, min_count=1,
#                                vector_size=100, window=5)


#print(model1)

