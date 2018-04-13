'''
This file generates TF-IDF values

@author: amanul
'''

from __future__ import print_function
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from string import punctuation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from pprint import pprint
from string import punctuation
	
#Method that generates tf-idf values for the comments
def generate_tf_idf(file):
	print("Generating Tf-Idf np arrays ...")
	vectorizer = TfidfVectorizer(ngram_range=(1,1))
	
	#output files for the np-arrays generated
	output_files = ['Data/tfidf.npy','Data/labels.npy','Data/thread_id.npy','Data/comment_position.npy','Data/comments.npy']
	count_vect = CountVectorizer()
	
	#Reading the data from preprocessed data file
	data = pd.read_csv(file, encoding = 'utf-8')
	label = np.array(data.Tag)
	thread = np.array(data.Comment_thread_id)
	comment_pos = np.array(data.comment_position)
	corpus = list((data.comments).values.astype('U'))
	
	#Initializing Vectorizer and then calculating tf-idf
	vectorizer = CountVectorizer(stop_words='english')
	X = vectorizer.fit_transform(corpus)
	tf = X.toarray()
	terms_index = vectorizer.get_feature_names()
	transformer = TfidfTransformer()
	Y = transformer.fit_transform(tf)
	tfidf = Y.toarray()
	
	#Saving all the np-arrays generated for later use
	np.save(output_files[0],tfidf)
	np.save(output_files[1],label)
	np.save(output_files[2],thread)
	np.save(output_files[3],comment_pos)
	np.save(output_files[4],corpus)
	
	print("TF-IDF files successfully generated")
	
	#Returning the list of np-arrays file namess
	return output_files

