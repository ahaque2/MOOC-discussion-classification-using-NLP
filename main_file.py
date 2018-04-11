import numpy as np
from Lematization import Lematize
from stemming import stemming
from tfidf_exp import generate_tf_idf
from classification import classification

Lematized_file = Lematize('Data/comments.csv')
stemmed_lematized_file = stemming(Lematized_file)
np_arrays = generate_tf_idf(Lematized_file)

tags = ['Q','O','B','BQ','C','S','P','T']
sk_fold = 10
tfidf = np.load(np_arrays[0])
label = np.load(np_arrays[1])
thread = np.load(np_arrays[2])
cpmment_pos = np.load(np_arrays[3])
comments = np.load(np_arrays[4])

algo = ['SVM']
#algo = []
for alg in algo:
	classification(tfidf, label, comments, tags, sk_fold, alg, 'rbf')
	classification(tfidf, label, comments, tags, sk_fold, alg, 'poly')
	classification(tfidf, label, comments, tags, sk_fold, alg, 'sigmoid')	
	classification(tfidf, label, comments, tags, sk_fold, alg, 'precomputed')