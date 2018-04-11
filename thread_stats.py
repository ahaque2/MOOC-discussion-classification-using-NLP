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
import pandas as pd
import matplotlib.pyplot as plt


def count_num_thread(thread):
		
	i = 0
	cmt_id = thread[0]
	for id in thread:
		if(cmt_id != id):
			cmt_id = id
			#print(i, "  ", cmt_id)
			i+=1
		
	return i

def find_tag(tag, thread, label):
	
	tags_per_thread = [0 for x in range(len(tag))]
	thread_stats = []
	cmt_id = thread[0]
	i = 0
	for id in thread:
		if(cmt_id == id):
			tags_per_thread[tag.index(label[i])] += 1
			
		else:
			cmt_id = id
			#print(i, "  ", cmt_id)
			thread_stats.append(tags_per_thread)
			#tags_per_thread.clear()
			tags_per_thread = [0 for x in range(len(tag))]
			tags_per_thread[tag.index(label[i])] += 1
		i+=1
	thread_stats.append(tags_per_thread)
	return thread_stats
	

def get_per_tag_stats(tag, tags_per_thread):
	
	per_tag_stat = []
	tags_count = []
	for i in range(len(tag)):
		for tag_count in tags_per_thread:
			tags_count.append(tag_count[i])
		per_tag_stat.append(tags_count)
		tags_count = []
	return per_tag_stat
	
		
tag = ['Q','O','B','BQ','C','S','P','T']
data = pd.read_csv('Data/comments.csv')
date = list(data.Date)
label = list(data.Tag)
thread = list(data.Comment_thread_id)
comment_list = list(data.comments)

num_of_threads = count_num_thread(thread)
#print("number of threads - ", num_of_threads)

#tags_per_thread = [[0 for x in range(len(tag))] for y in range(num_of_threads)]
tags_per_thread = find_tag(tag,thread,label)
#pprint(tags_per_thread)

tags_count = get_per_tag_stats(tag, tags_per_thread)
i=0
for t in tags_count:
	print(tag[i]+" \t",t,"\n")
	i+=1
	
print(len(tags_count[0]))
for i in range(len(tag)):
	plt.plot(list(range(num_of_threads+1)),tags_count[i])
	plt.ylabel('Count of Tag - '+tag[id])
	plt.xlabel('comment Thread Number')
	plt.show()
		



			
				
			
