'''
This is the code for performing lemmatization on the given dataset

@author: amanul
'''

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
#from stop_words import get_stop_words
from string import punctuation

def Lematize(file):

	print("Performing Lemmatization ...")

	#Path for the output file to be dumped after Lemmatization
	output_file = 'Data/Lematized_comments.csv'
	flag_np = 0
	
	#Reading the data from the original file
	data = pd.read_csv(file)
	date = list(data.Date)
	label = list(data.Tag)
	thread = list(data.Comment_thread_id)
	comment_list = list(data.comments)
	comment_pos = list(data.comment_position)
	
	tokenized_comments_list = []
	pos_tagged_comments = []
	lemmatized_tokens = []
	lemmatized_comments = []
	
	i=0
	j=0
	#Using wordnetLemmatizer api for lemmatization
	wordnet_lemmatizer = WordNetLemmatizer()
	for cmt in comment_list:
		
		cmt = cmt.replace('\\n','')
		
		#Tokenizing the comments
		tokenized_comments = word_tokenize(cmt)
		tokenized_comments_list.append(tokenized_comments)
		
		#Tagging the tokens with appropriate POS (Part-of-Speech) Tag
		pos_tagged = nltk.pos_tag(tokenized_comments)
		pos_tagged_comments.append(pos_tagged)

	stop_word = stopwords.words('english') + list(punctuation)
	stop_words = set(stop_word)

	comment_string = ''
	for cmt in pos_tagged_comments:
		#constructing a comment string from nothing.
		comment_string = ''
		for l in cmt:
			word = l[0]
			tag = l[1]
		
			if(l[0].isalpha()):
				word = word.lower()
			
			#Using POS tags to correctly determine the corresponding Lemmatized word for each word in a comment.
			#Each word is tagged with one of the types of english grammar like - adjective, verb, noun etc based on which it is lematized
			if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
				tag = wordnet.VERB
			elif (tag == 'NN' or tag == 'NNS'):
				tag = wordnet.NOUN
			elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
				tag = wordnet.ADJ 
			elif tag == 'RB' or tag == 'RBR' or tag =='RBS':
				tag = wordnet.ADV
			elif (tag == 'NNP' or tag == 'NNPS') and flag_np == 1:
				tag = 'PN'
			elif (tag == 'NNP' or tag == 'NNPS') and flag_np == 0:
				tag = wordnet.NOUN
			else:
				tag = wordnet.NOUN
			
			if word not in stop_words and tag != 'PN':
				lemmatized_word = wordnet_lemmatizer.lemmatize(word,pos=tag)
				comment_string = comment_string + " " + lemmatized_word
			
		comment_string
		lemmatized_comments.append(comment_string)

	i = 0
	#Storing the lematized comments along with other attributes in a csv output file
	out = open(output_file, 'w', newline='', encoding='utf8')
	fieldnames = ['Date', 'Comment_thread_id','comment_position','Tag', 'comments']
	writer = csv.DictWriter(out,fieldnames)
	writer.writerow({'Date': 'Date', 'Comment_thread_id':'Comment_thread_id', 'comment_position':'comment_position', 'Tag':'Tag','comments':'comments'})

	#Iterating through all the Lematized comments to store in a new csv file
	for cmt in lemmatized_comments:
		
		cmt = cmt.replace('\\n','')
		writer.writerow({'Date': date[i], 'Comment_thread_id':thread[i], 'comment_position':comment_pos[i],'Tag':label[i],'comments':cmt})
		i = i + 1

	print("Lemmatization successfully done")
	
	#Return the csv file name that has the processed data
	return output_file
		

				
					
				
