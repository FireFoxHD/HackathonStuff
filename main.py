import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.offline as py
# import plotly.graph_objs as go
# import plotly.tools as tls
# import plotly.express as px
import string
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score


##############################~RAW DATA~#####################################################################################
col_names = ['headingText', 'emotion']
df = pd.read_csv('everything.csv', header=None, names=col_names)
df = df.iloc[1: , :]
# print(df.head())
# fig = px.histogram(df, x="emotion")
# fig.show()


##############################~PRE-PROCESSING~###############################################################################
#Assigning numerical values to the diff emotions
df = df[df['emotion'] !="-"] #drop heading texts with '-' emotional value
df['emotionVal'] = df['emotion'].apply(lambda rating: 0 if rating == "n" else +1) #if emotion == "n" give it +1 and if it is "p" call it -1
#print(df.head())
neutral= df[df['emotionVal'] == 0]
panic = df[df['emotionVal'] == 1]

#APPLYING ALL FUNCTIONS ALL AT ONCE
stopWords = nltk.corpus.stopwords.words('english') #get the stopwords in english
wn = nltk.WordNetLemmatizer()
def clean_up(text):
    noPunctText = "".join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', noPunctText) #split on all non-word characters
    noStopWords = [word for word in tokens if word not in stopWords]
    text_nsw = [each_string.lower() for each_string in noStopWords]
    lem_text = [wn.lemmatize(word) for word in text_nsw]
    return lem_text


##############################~HEARTBREAK ANNIVERSARY (BREAK UP INTO TESTING VS TRAINING DATA)~##################################
df['headingTextClean'] = df['headingText'].apply(clean_up)
dfNew = df[['headingTextClean', 'emotionVal']]
dfNew.drop(['emotionVal'], axis=1)


heading_dummies = pd.get_dummies(dfNew['headingTextClean'].apply(tuple).value_counts(), drop_first=True)
heading_dummies.reset_index(drop=True, inplace=True)
dfNew.reset_index(drop=True, inplace=True)
dfNew = pd.concat([dfNew, heading_dummies],axis=1)
dfNew = dfNew.dropna(axis=1)

X = dfNew.drop(['headingTextClean'], axis=1)
y = dfNew['emotionVal']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=5)

model = MultinomialNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.predict([["Oh my gosh my period just came! What do I do?"]]))

# svc_f1 = f1_score(y_test, model.predict(X_test), average=None, labels=['neutral', 'panic'])
# print("F1 score of SVC: ", svc_f1)





