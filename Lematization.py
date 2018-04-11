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

	output_file = 'Data/Lematized_comments.csv'
	
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
	wordnet_lemmatizer = WordNetLemmatizer()
	for cmt in comment_list:
		
		cmt = cmt.replace('\\n','')
		tokenized_comments = word_tokenize(cmt)
		tokenized_comments_list.append(tokenized_comments)
		pos_tagged = nltk.pos_tag(tokenized_comments)
		pos_tagged_comments.append(pos_tagged)

	stop_word = stopwords.words('english') + list(punctuation)
	stop_words = set(stop_word)
	token_set = set()
	comment_string = ''
	for cmt in pos_tagged_comments:
		comment_string = ''
		for l in cmt:
			word = l[0]
			tag = l[1]
		
			if(l[0].isalpha()):
				word = word.lower()
				
			token_set.add(l[1])
			if tag == 'VB' or tag == 'VBD' or tag == 'VBG' or tag == 'VBN' or tag == 'VBP' or tag == 'VBZ':
				tag = wordnet.VERB
			elif tag == 'NN' or tag == 'NNS' or tag == 'NNP' or tag == 'NNPS':
				tag = wordnet.NOUN
			elif tag == 'JJ' or tag == 'JJR' or tag == 'JJS':
				tag = wordnet.ADJ 
			elif tag == 'RB' or tag == 'RBR' or tag =='RBS':
				tag = wordnet.ADV
			elif tag == 'NNP':
				tag = 'PN_1'
			elif tag == 'NNPS':
				tag = 'PN_2'
			else:
				tag = wordnet.NOUN
			#print(l[1])
			if word not in stop_words and tag != 'PN_1' and tag != 'PN_2':
				lemmatized_word = wordnet_lemmatizer.lemmatize(word,pos=tag)
				#set_lemmatized.add(lemmatized_word)
				comment_string = comment_string + " " + lemmatized_word
			elif word not in stop_words and tag == 'PN_1':
				nnp.add(word)
			elif word not in stop_words and tag == 'PN_2':
				nnps.add(word)
		comment_string
		lemmatized_comments.append(comment_string)

	i = 0
	out = open(output_file, 'w', newline='', encoding='utf8')
	fieldnames = ['Date', 'Comment_thread_id','comment_position','Tag', 'comments']
	writer = csv.DictWriter(out,fieldnames)
	writer.writerow({'Date': 'Date', 'Comment_thread_id':'Comment_thread_id', 'comment_position':'comment_position', 'Tag':'Tag','comments':'comments'})

	for cmt in lemmatized_comments:
		
		cmt = cmt.replace('\\n','')
		writer.writerow({'Date': date[i], 'Comment_thread_id':thread[i], 'comment_position':comment_pos[i],'Tag':label[i],'comments':cmt})
		i = i + 1

	print("Lemmatization successfully done")
	
	return output_file
		

				
					
				
