'''
This is the code to get class distribution stats for the dataset being used

@author: amanul
'''

import csv
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

def distribution(file, tags):

	#Reading Data from csv file
	data = pd.read_csv(file)
	label = list(data.Tag)
	
	#Initializing an initial distribution of 0 for all labels
	distribution = [0 for x in range(len(tags))]
	print(distribution)
	
	for t in label:
		#Updating the distribution by incrementing with 1 for each label as it appears in dataset
		distribution[tags.index(t)] += 1
	
	print(distribution)
	print(sum(distribution))
	
	color = ['yellow','lawngreen','lightskyblue','orangered','darkgrey','coral','teal','firebrick']
	
	#Ploting a pie chart for distribution across all labels
	plt.pie(distribution, labels=tags, colors = color, autopct='%1.1f%%', shadow=False, startangle=140)
	plt.axis('equal')
	plt.show()

#Classes for classification that are present as Tag for comments in the dataset
tags = ['Q','O','B','BQ','C','S','P','T']
distribution('Data/comments.csv', tags)

	