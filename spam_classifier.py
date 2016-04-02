import pandas as pd
import csv
import matplotlib.pyplot as plt
import cPickle
import numpy as np
import time

from textblob import TextBlob

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 

#the data for the code below is available on the following link
# https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

# here we use naive bayes and svm for sms classification
# the following is adapted from tutorial on http://radimrehurek.com/data_science_python/
# it contains some additional thoughts and analysis apart from tutorial

#function to split the mssges and convert to bag of words representation and
#1). remove stopwords like is,an etc as these words carry very less information about our target class.
#2). getting the morphological root of a word since going, for most purposes,  gives same information as go gives. 
#3). Cases dont enhance the information given by a word. So converting all the messages to lowercase makes sense.
#    All of the above helps to achieve reduced dimensionality. Since as dimensionality increases theamount
#    data needed for getting a model that generalises well increases. This is not generally possible. So 
#    techiques like the above are used, since text classification with bag of words representation has high
#    high dimensionality.

#use nltk corpus to get the stopwords
def convert(data):
	data = unicode(data,"utf-8").lower()
	data = TextBlob(data).words
	stop = stopwords.words('english')
	data = list(set(data)-set(stop))

	return [word.lemma for word in data]

def convert_direct(data):
	data = unicode(data,"utf-8").lower()

	return TextBlob(data).words

#getting the feel of data
data=[line.strip() for line in open('/home/rishab/ML/cs229/post_collegeate_earning/preprocess_experiments/smsspamcollection/SMSSpamCollection')]

print len(data)
for msg_no, msg in enumerate(data[:5]):
	print msg_no,":",msg

#using pandas to get statistics of the data
df=pd.read_csv("/home/rishab/ML/cs229/post_collegeate_earning/preprocess_experiments/smsspamcollection/SMSSpamCollection", sep="\t", names=["labels","mssg"], quoting=csv.QUOTE_NONE)
#check data
print df.head()
#statistics
print df.groupby("labels").describe()
# getting the length of mssg which could be conisdered as an important feaure for filtering
df['len_of_mssg']=df['mssg'].map(lambda text : len(text))
print df.head()
print df.mssg.shape,df.labels.shape

print df.len_of_mssg.describe()

#bag of words representation of data
bow=df.mssg.apply(convert)
print bow.head()
print TextBlob("this is a test sentence").tags

#difference of direct and conversion after stopwords removal
print df.mssg.head().apply(convert)
print df.mssg.head().apply(convert_direct)

#  converting to form for making a ml model

bow_transformer = CountVectorizer(analyzer=convert).fit(df.mssg)
# print bow_transformer.vocabulary_
# print df.mssg[2]
# print convert(df.mssg[2])
# print bow_transformer.transform([df.mssg[2]])
mssg_to_bow = bow_transformer.transform(df.mssg)
test_msg=bow_transformer.transform(["festive offer"])

# print mssg_to_bow.nnz
# print mssg_to_bow
print "sparsity : ",(mssg_to_bow.nnz/float(mssg_to_bow.shape[0]*mssg_to_bow.shape[1]))*100.0

#using the tfidf for document weight and normalisation

tfidf_transformer = TfidfTransformer().fit(mssg_to_bow)
print tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]
data_tfidf = tfidf_transformer.transform(mssg_to_bow)
test_tfidf=tfidf_transformer.transform(test_msg)
#getting the tme for training the model
millis = int(round(time.time() * 1000))
#building model
#naive bayes
print df.labels.head()
model = MultinomialNB().fit(data_tfidf,np.asarray(df['labels'],dtype="|S6"))
print model.predict(test_tfidf)
test_msgs=["Diwali bumper sale","Festive offer"]
#when we run prediction on festive offer then we get a misclassification. This proves that classifier
#is as good as the trainig data especially in case of text classification. This is the case for all examples in 
#test_msgs. Classifier is not able to predict well for thise words in spam that it doesnt see in 
#the training data as we see festive, offer etc are not there in the vocabulary
try:
	print tfidf_transformer.idf_[bow_transformer.vocabulary_['festive']]
except Exception, e:
	print "not there in vocabulay of the CountVectorizer"

#Experimenting with the data
#getting the test_train_split of the for getting the data for validation/testing
msg_train,msg_test,msg_label_train,msg_label_test = train_test_split(df.mssg,df.labels,test_size=0.3)
print msg_label_train.shape,msg_train.shape
#creating the model using pipeline
pipeline=Pipeline([
	('bow',CountVectorizer()),
	('tfidf',TfidfTransformer()),
	('classifier',MultinomialNB())])

scores=cross_val_score(pipeline, msg_train, np.asarray(msg_label_train,dtype="|S6"), cv=5, scoring="accuracy")
print scores

using gridsearch to get the optmized parmeters
params={
	'tfidf__use_idf': (True, False),
	"bow__analyzer" : (convert,convert_direct)
}
print "startified",StratifiedKFold(msg_label_train,n_folds=5)
grid=GridSearchCV(
	pipeline,
	params,
	refit = True,
	scoring='accuracy',
	cv=StratifiedKFold(msg_label_train,n_folds=5)
	)

spam_gs = grid.fit(msg_train,np.asarray(msg_label_train,dtype="|S6"))
print spam_gs.grid_scores_

# performing gridsearch for svm model
# Grid Search is a great class to efficiently run algorithm many times for tuning optimal model para-
# meters
pipeline_svm=Pipeline([
	('bow' , CountVectorizer(analyzer=convert)),
	('tfidf' , TfidfTransformer()),
	('classifier' , SVC())
	])
params_svm = [
{
	'classifier__C' : [1,10,100,1000],
	'classifier__kernel' : ['linear']
},
{
	'classifier__C' : [1,10,100,1000],
	'classifier__kernel' : ['rbf'],
	'classifier__gamma' : [0.001,0.001,0.01]	
}
]
grid_svm = GridSearchCV(
	pipeline_svm,
	param_grid=params_svm,
	refit=True,
	scoring='accuracy',
	cv=StratifiedKFold(msg_label_train,n_folds=5)
	)
svm_results=grid_svm.fit(msg_train,np.asarray(msg_label_train,dtype="|S6"))
print svm_results.grid_scores_

#storing the model params
with open("sms_predictor_model.pkl",'wb') as fout:
	cPickle.dump(svm_results,fout)

# When we run the grid search for spam detection , it is observed that naive bayes runs faster but 
# the accuracy observed is much lower around 0.95 as compared to >0.98 in case of SVC. This is primari;y
# because naive bayes is a fairly simple model while SVC can have a fairly complex model.
