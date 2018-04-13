'''
This file Generates comment thread stats and plot for each labels per comment thread

@author: amanul
'''

import re
import csv
import pandas as pds
import matplotlib.pyplot as plt

#Method that returns the number of threads of comments in the dataset
def count_num_thread(thread):
	
	i = 0
	cmt_id = thread[0]
	for id in thread:
		if(cmt_id != id):
			cmt_id = id
			#print(i, "  ", cmt_id)
			i+=1
		
	return i

#Method that returns the number of labels in each comment_thread
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
			thread_stats.append(tags_per_thread)
			tags_per_thread = [0 for x in range(len(tag))]
			tags_per_thread[tag.index(label[i])] += 1
		i+=1
	thread_stats.append(tags_per_thread)
	return thread_stats
	
#Method for generating per comment thread distribution for each label
def get_per_tag_stats(tag, tags_per_thread):
	
	per_tag_stat = []
	tags_count = []
	for i in range(len(tag)):
		for tag_count in tags_per_thread:
			tags_count.append(tag_count[i])
		per_tag_stat.append(tags_count)
		tags_count = []
	return per_tag_stat
			
#Classes for classification
tag = ['Q','O','B','BQ','C','S','P','T']
data = pd.read_csv('Data/comments.csv')
date = list(data.Date)
label = list(data.Tag)
thread = list(data.Comment_thread_id)
comment_list = list(data.comments)

num_of_threads = count_num_thread(thread)

tags_per_thread = find_tag(tag,thread,label)

tags_count = get_per_tag_stats(tag, tags_per_thread)
i=0
for t in tags_count:
	print(tag[i]+" \t",t,"\n")
	i+=1
	
print(len(tags_count[0]))

#Generating plot for each label per comment thread distribution
for i in range(len(tag)):
	plt.plot(list(range(num_of_threads+1)),tags_count[i])
	plt.ylabel('Count of Tag - '+str(tag[i]))
	plt.xlabel('comment Thread Number')
	plt.show()
		



			
				
			
