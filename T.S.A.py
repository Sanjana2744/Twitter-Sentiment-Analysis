import pandas as pd 
import numpy as np 
import re 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print (stopwords.words('english'))

# Loading data from csv to pandas dataframe 
twitter_data = pd.read_csv("twitter_dataset.csv", encoding = 'ISO-8859-1')

# Checking numbers of rows and columns
twitter_data.shape

# printing first five rows
twitter_data.head()

twitter_data.isnull().sum()

twitter_data.dropna(subset = ['tweet','sentiment'], inplace = True)
twitter_data.isnull().sum()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'@[a-zA-Z0-9_]+','',text)
    text = re.sub(r'#', '',text)
    text = re.sub(r'[^a-zA-Z\S]','',text)
    return text

twitter_data['Tweet_text'] = twitter_data['tweet'].apply(preprocess_text)

twitter_data.head(4)

count_vectorizer = CountVectorizer(max_features = 5000)
count_matrix = count_vectorizer.fit_transform(twitter_data['Tweet_text'])

X_train, X_test, Y_train, Y_test = train_test_split(count_matrix, twitter_data['sentiment'], test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train,Y_train)

y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:",accuracy)

with open('count_vectorizer.pkl','wb') as vectorizer_file:
    pickle.dump(count_vectorizer, vectorizer_file)

with open('nb_classifier.pkl', 'wb') as classifier_file:
    pickle.dump(nb_classifier, classifier_file) 
