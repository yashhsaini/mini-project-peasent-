import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
import time  
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 
from nltk.tokenize import RegexpTokenizer   
from nltk.stem.snowball import SnowballStemmer 
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.pipeline import make_pipeline 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import pickle 
import warnings 
warnings.filterwarnings('ignore')
phish_data = pd.read_csv('phishing_site_urls.csv')
phish_data.head()
phish_data.tail()
phish_data.info()
phish_data.isnull().sum()
label_counts = pd.DataFrame(phish_data.Label.value_counts())
fig = px.bar(label_counts, x=label_counts.index, y=label_counts.Label)
fig.show()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
phish_data.URL[0]
tokenizer.tokenize(phish_data.URL[0])
print('Getting words tokenized ...')
t0= time.perf_counter()
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) 
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')
phish_data.sample(5)
stemmer = SnowballStemmer("english")
print('Getting words stemmed ...')
t0= time.perf_counter()
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')
phish_data.sample(5)
print(' joiningwords ...')
t0= time.perf_counter()
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')
phish_data.sample(5)
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']
bad_sites.head()
good_sites.head()
data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)
data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)
cv = CountVectorizer()
feature = cv.fit_transform(phish_data.text_sent)
feature[:5].toarray()
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)
lr = LogisticRegression()
lr.fit(trainX,trainY)
lr.score(testX,testY)
Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)
print('\n For Logistic regression')
print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,target_names =['Bad','Good']))
print('\n')
mnb = MultinomialNB()
mnb.fit(trainX,trainY)
mnb.score(testX,testY)
Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)
print('\nFor multinomialNB')
print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY, target_names =['Bad','Good']))
print('\n')
pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)
pipeline_ls.fit(trainX,trainY)
pipeline_ls.score(testX,testY) 
print('\n final model report')
print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY, target_names =['Bad','Good']))
pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)
predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)