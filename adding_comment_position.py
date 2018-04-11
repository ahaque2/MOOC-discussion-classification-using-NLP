import re
import csv
import pandas as pd
from pprint import pprint
from datetime import datetime
import operator
import nltk
from nltk import FreqDist
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from string import punctuation

data = pd.read_csv('3_Stemmed_Lematized_comments_AM_removed.csv')
date = list(data.Date)
label = list(data.Tag)
thread = list(data.Comment_thread_id)
comment_list = list(data.comments)
comment_position = []

cmt_id = thread[0]
j = 1
for id in thread:
	if(cmt_id == id):
		comment_position.append(j)
		j = j + 1
	else:
		cmt_id = id
		comment_position.append(1)
		j = 2

out = open('3_Stemmed_Lematized_comments_AM_removed_with_comment_position.csv', 'w', newline='', encoding='utf8')
fieldnames = ['Date', 'Comment_thread_id', 'comment_position','Tag', 'comments']
writer = csv.DictWriter(out,fieldnames)
writer.writerow({'Date': 'Date', 'Comment_thread_id':'Comment_thread_id','comment_position':'comment_position', 'Tag':'Tag','comments':'comments'})

i = 0
for cmt in comment_list:
	
	#labels=np.load('label.npy')
	writer.writerow({'Date': date[i], 'Comment_thread_id':thread[i], 'comment_position':comment_position[i],'Tag':label[i],'comments':cmt})
		#print(comments[i], "  ",label[i],"  ", predicted_label)
	print(cmt)
	i = i + 1
	
#print(len(set_original))
#print(len(set_lemmatized))


#print(token_set)
	

			
				
			
