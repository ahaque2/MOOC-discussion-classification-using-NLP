import csv
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

def distribution(file, tags):

	data = pd.read_csv(file)
	label = list(data.Tag)
	distribution = [0 for x in range(len(tags))]
	print(distribution)
	
	for t in label:
		distribution[tags.index(t)] += 1
	
	print(distribution)
	print(sum(distribution))
	
	color = ['yellow','lawngreen','lightskyblue','orangered','darkgrey','coral','teal','firebrick']
	#'pink','c','orange']
	
	plt.pie(distribution, labels=tags, colors = color, autopct='%1.1f%%', shadow=False, startangle=140)
	plt.axis('equal')
	plt.show()
	
tags = ['Q','O','B','BQ','C','S','P','T']
tags = ['Q','O','B','BQ','A','M','C','CQ','S','P','T']
distribution('Data/Original_Dataset.csv', tags)

	