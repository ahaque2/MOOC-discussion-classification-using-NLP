from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import math
import csv
import pandas as pd
from pprint import pprint
from datetime import datetime
import operator
import nltk
from string import punctuation

def stemming(file):
	
	print("Performing Stemming ...")
	ps = PorterStemmer()
	output_file = 'Data/stemmed_Lemetized_comments.csv'	
	data = pd.read_csv(file)
	date = list(data.Date)
	label = list(data.Tag)
	thread = list(data.Comment_thread_id)
	comment_list = list(data.comments)
	comment_pos = list(data.comment_position)
	tokenized_comments_list = []
	pos_tagged_comments = []
	stemmed_tokens = []
	stemmed_comments = []
	tokenized_comments = []

	for cmt in comment_list:
		comment = ''
		if(str(cmt) == 'nan'):
			tokenized_comments = []
		else:
			tokenized_comments = word_tokenize(cmt)
		for word in tokenized_comments:
			stemmed_word = ps.stem(word)
			comment = comment + " " + stemmed_word
		stemmed_comments.append(comment)
		tokenized_comments.clear()
		
	i = 0
	out = open(output_file, 'w', newline='', encoding='utf8')
	fieldnames = ['Date', 'Comment_thread_id','comment_position','Tag', 'comments']
	writer = csv.DictWriter(out,fieldnames)
	writer.writerow({'Date': 'Date', 'Comment_thread_id':'Comment_thread_id','comment_position':'comment_position', 'Tag':'Tag','comments':'comments'})

	for cmt in stemmed_comments:
		
		cmt = cmt.replace('\\n','')
		writer.writerow({'Date': date[i], 'Comment_thread_id':thread[i], 'comment_position':comment_pos[i],'Tag':label[i],'comments':cmt})
		i = i + 1
		
	print("Stemming successfully done")
		
	return output_file