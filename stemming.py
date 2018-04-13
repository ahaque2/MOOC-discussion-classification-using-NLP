'''
This is the code for performing stemming on the comments

@author: amanul
'''

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

#Method that does stemming of comments
def stemming(file):
	
	print("Performing Stemming ...")
	
	#Using PorterStemmer from nltk for stemming
	ps = PorterStemmer()
	
	#Path for output file
	output_file = 'Data/stemmed_Lemetized_comments.csv'	
	
	#Reading data from input file
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

	#Iterating through each comment one by one and stemming
	for cmt in comment_list:
		comment = ''
		if(str(cmt) == 'nan'):
			tokenized_comments = []
		else:
			tokenized_comments = word_tokenize(cmt)
			
		for word in tokenized_comments:
			#Stemming the tokenized words and appending to a new string comment
			stemmed_word = ps.stem(word)
			comment = comment + " " + stemmed_word
			
		#Appending the stemmed comment to processed comments list
		stemmed_comments.append(comment)
		tokenized_comments.clear()
		
	i = 0
	#Output file creation to store the stemmed comments
	out = open(output_file, 'w', newline='', encoding='utf8')
	fieldnames = ['Date', 'Comment_thread_id','comment_position','Tag', 'comments']
	writer = csv.DictWriter(out,fieldnames)
	writer.writerow({'Date': 'Date', 'Comment_thread_id':'Comment_thread_id','comment_position':'comment_position', 'Tag':'Tag','comments':'comments'})

	#Iterating through the stemmed comments and storying it into the output csv file
	for cmt in stemmed_comments:
		
		cmt = cmt.replace('\\n','')
		writer.writerow({'Date': date[i], 'Comment_thread_id':thread[i], 'comment_position':comment_pos[i],'Tag':label[i],'comments':cmt})
		i = i + 1
		
	print("Stemming successfully done")
		
	#Return the file_name of the output file
	return output_file