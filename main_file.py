'''
This is the master file that starts the Data-Prepossessing and then calls the classification algorithms for training and testing on the processed Data 

@author: amanul
'''

import numpy as np
from Lematization import Lematize
from stemming import stemming
from tfidf import generate_tf_idf
from classification import classification

#Lematizing the comments of original Dataset
Lematized_file = Lematize('Data/comments.csv')

#Stemming the Lematized comments from previous step
stemmed_lematized_file = stemming(Lematized_file)

#Generating tf-idf np-array for the preprocessed data
np_arrays = generate_tf_idf(Lematized_file)

#Classes for classification of each comment. Each comment is classified into one of these classes.
tags = ['Q','O','B','BQ','C','S','P','T']

#Number of folds for the k-fold validation
sk_fold = 10

#Loading required data from the stored np-arrays
tfidf = np.load(np_arrays[0])
label = np.load(np_arrays[1])
thread = np.load(np_arrays[2])
comment_pos = np.load(np_arrays[3])
comments = np.load(np_arrays[4])

#List of algorithm that needs to be run on the given data
algo = ['DT','NN','KNN','NB','RF','ERF','ABC','GBC','SVM','ensemble']
for alg in algo:
	#Calling the classification.py file that does the remaining tasks of training and testing the various models and prints the results
	classification(tfidf, label, comments, tags, sk_fold, alg)


