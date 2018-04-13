'''
This file trains te classifier models using the tf-idf values generated.
Then uses cross-validation by splitting the given data to compute results metrics like - accuracy, precision, recall and f1-scores.

@author: amanul
'''

import pandas as pd
import numpy as np
import csv
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from pprint import pprint
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 


#Method for training and testing classification models
def classification(tfidf, label, comments, tag, sk_fold, algo):
	i = 0
	models = []
	predictions = []
	precision = []
	recall = []
	f_measure = []
	total_precision = []
	total_recall = []
	total_f_measure = []
	accuracies = []
	test_x = []
	test_y = []
	testing_index = []
	
	print("Results using classification algorithm ", algo)
	
	#Sk-fold validation to split the data into test and train data
	kf = model_selection.StratifiedKFold(n_splits = sk_fold, random_state=0, shuffle = False)

	for train_index, test_index in kf.split(tfidf,label):
		
		#Splitting test and train data
		x_train, x_test = tfidf[train_index], tfidf[test_index]
		y_train, y_test = label[train_index], label[test_index]
		
		#List for various models, predictions and results from different k-fold runs
		clf_list = []
		predict_list = []
		results = []
		
		#Using scaler fit to improve training data for Neural Network
		if(algo=='NN'):
			scaler = StandardScaler()
			scaler.fit(x_train)  
			x_train = scaler.transform(x_train)  
			x_test = scaler.transform(x_test) 
			

		#Training the model with the specified classifier
		clf_list, score, tag_clf = train_model(x_train, y_train, algo)

		clf_list, predict_list, score, tag_clf = train_model_2(x_train, y_train, algo)

		models.append(clf_list)
		#joblib.dump(clf_list, str(i)+'_model_stemmed_Lem_3_20_iter.pkl')
		
		test_x.append(x_test)
		test_y.append(y_test)
		testing_index.append(test_index)
		
		#Getting results from by testing the model using above trained classifier model
		results, tag_probability = test_model(x_test, y_test, clf_list)
		
		#Getting metrics (accuracy, precision, recall, f1-score and confusion matrix) from the results obtained
		acc, pr, r, f, confusion_matrix, class_report = compute_metric(results, test_index, label, tag_clf)
		prec, rec, f_meas, support = class_report_(class_report, tag)
		
		#Appending the results for each k-fold iteration to be used to average the results and obtain a final overall results
		total_precision.append(pr)
		total_recall.append(r)
		accuracies.append(acc)
		total_f_measure.append(f)
		
		precision.append(prec)
		recall.append(rec)
		f_measure.append(f_meas)
		
		#Printing the results for each iterations
		print("Start --------- ", i)
		print("number of training data ", len(train_index), "  Number of test data instances", len(test_index))
		
		print("Suuport for each class - \t", support)
		print("precision - ", pr, "\t\t", prec)
		print("recall - ", r, "\t\t\t", rec)
		print("f1-measure - \t\t\t", f_meas)
		print("Accuracy - ", acc)
		print("Confusion Matrix - \n" , confusion_matrix)
		print("Ends here -----------\n")
		
		i = i+1
	
	#objects to store the averaged results from k-fold iterations
	avg_precision = [0 for x in range(len(tag))]
	avg_recall = [0 for x in range(len(tag))]
	avg_f_measure = [0 for x in range(len(tag))]
	
	
	for u in range(len(tag)):
		for v in range(sk_fold):
			avg_precision[u] = avg_precision[u] + precision[v][u]
			avg_recall[u] = avg_recall[u] + recall[v][u]
			avg_f_measure[u] = avg_f_measure[u] + f_measure[v][u]
		avg_precision[u] = round(avg_precision[u]/sk_fold,2)
		avg_recall[u] = round(avg_recall[u]/sk_fold,2)
		avg_f_measure[u] = round(avg_f_measure[u]/sk_fold,2)
		

	#Writing the final results after averaging all values from k-fold iterations into text files
	f= open("Results/Results_/Results_"+str(algo)+"_"+str(sk_fold)+"_.txt","w+")

	f= open("Results/Results_removed_PN/Results_"+str(algo)+"_"+str(sk_fold)+"_.txt","w+")

	f.write("precision - " + str(sum(total_precision)/sk_fold)+ "\t"+ str(avg_precision) +"\n")
	f.write("recall - " + str(sum(total_recall)/sk_fold) + "\t\t" + str(avg_recall)+"\n")
	f.write("f1-measure - " + str(sum(total_f_measure)/sk_fold) + "\t" + str(avg_f_measure)+"\n")
	f.write("Accuracy - " + str(sum(accuracies)/sk_fold)+"\n")
	
	#Writing the final results after averaging all values from k-fold iterations into text files
	print("\n\nFinal results for this case ----------")
	print("precision - ", sum(total_precision)/sk_fold, "\t", avg_precision)
	print("recall - ", sum(total_recall)/sk_fold, "\t", avg_recall)
	print("f1-measure - ", sum(total_f_measure)/sk_fold, "\t",  avg_f_measure)
	print("Accuracy - ", sum(accuracies)/sk_fold)
	print("\n")

#Method to declare the classifier depending on the algo	
def ml_(x, y, algo):
	# x is vector, y is label, algo is the algorithm to be used
	#Default algo is SVM
	clf = svm.SVC(kernel='linear',probability=True)
	if(algo == 'NN'):
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000,), random_state=1)
	elif(algo == 'DT'):
		clf = tree.DecisionTreeClassifier()
	elif(algo == 'KNN'):
		clf = KNeighborsClassifier(n_neighbors=51)
	elif(algo == 'NB'):
		clf = GaussianNB()
	elif(algo == 'RF'):
		clf = RandomForestClassifier(n_estimators=10)
	elif(algo == 'ETC'):
		clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
	elif(algo == 'ABC'):
		clf = AdaBoostClassifier(n_estimators=100)
	elif(algo == 'GBC'):
		clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	elif(algo == 'GBR'):
		clf = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
	
	#ensemble method that combines 3 classifiers with soft voting schemes
	elif(algo == 'ensemble'):
		clf1 = KNeighborsClassifier(n_neighbors=35)
		clf2 = svm.SVC(kernel='linear',probability=True)
		clf3 = RandomForestClassifier(n_estimators=10)
		clf = VotingClassifier(estimators=[('knn', clf1), ('svc', clf2), ('rf', clf3)], voting='soft', weights=[2,3,2])
		
	clf.fit(x, y)
	tag_clf = clf.classes_
	predict = clf.predict(x)
	score = cross_val_score(clf, x, y)
	
	#return fitted classifier, predicted classes, score and list of tag (in order as used by the classifier) 
	return clf, predict, score, tag_clf


#Method to train the model on the given training data
def train_model(data, y, algo):
	
	clf_list = []
	label_list = []
	print('Train model')
	label = y.tolist()

	clf, predict, score, tag_clf = ml_(data, label, algo)
	clf_list.append(clf)
	
	#return trained model (clf_list), predicted class list(predict_list), score and tags/class list
	return clf_list, score, tag_clf

#Method to test the trained model using unseen test data
def test_model(data, y, clf_list):
	
	predict_list_results = []
	predict_probability_list = []
	
	for clf in clf_list:
		predict = clf.predict(data)
		predict_probability = clf.predict_proba(data)
		predict_probability_list.append(predict_probability)
		predict_list_results = predict.tolist()
	
	#Return the predicted class list and probability list for predicted list
	return predict_list_results, predict_probability_list
	
#Method to compute the result's metrics
def compute_metric(results, test_index, label, tag):

	actual_label = []
	predicted_label = []
	for predicted, i in zip(results, test_index):	
		actual_label.append(label[i])
		predicted_label.append(predicted)
		
	#Computing various metrics for te results obtained from testing the model
	accuracy = round(accuracy_score(actual_label,predicted_label),2)
	precision = round(precision_score(actual_label,predicted_label,average='weighted'),2)
	recall = round(recall_score(actual_label,predicted_label, average='weighted'),2)
	f_measure = round(f1_score(actual_label,predicted_label, average='weighted'),2)
	
	#Printing the results and class report on the console
	print("recall - ",round(recall_score(actual_label,predicted_label, average='macro'),2))
	print("precision - ", round(precision_score(actual_label,predicted_label,average='macro'),2))
	class_report = classification_report(actual_label, predicted_label, target_names=tag)
	print(class_report)
	classif_report = classifaction_report(class_report)
	confusion_mat = confusion_matrix(actual_label, predicted_label)
	
	#Returning metrics, confusion metrics and class report
	return accuracy, precision, recall, f_measure, confusion_mat, classif_report
	
#Method to convert the text classification report into Panda dataframe
def classifaction_report(report):

	report_data = []
	lines = report.split('\n')
	for line in lines[2:-3]:
		row = {}
		row_data = line.split('      ')
		row['class'] = row_data[1]
		row['precision'] = float(row_data[2])
		row['recall'] = float(row_data[3])
		row['f1_score'] = float(row_data[4])
		row['support'] = float(row_data[5])
		report_data.append(row)
		dataframe = pd.DataFrame.from_dict(report_data)
	
	return dataframe
	
#Method to compute class-wise metrics 
def class_report_(class_report, tag):
	
	prec = [0 for x in range(len(tag))]
	rec = [0 for x in range(len(tag))]
	f_meas = [0 for x in range(len(tag))]
	support = [0 for x in range(len(tag))]
	
	for l in range(len(tag)):
		e = tag.index(class_report['class'][l].strip())
		prec[e] = round(float(class_report['precision'][l]),2)
		rec[e] = round(float(class_report['recall'][l]),2)
		f_meas[e] = round(float(class_report['f1_score'][l]),2)
		support[e] = round(float(class_report['support'][l]),0)
		
	#Return the results for class-wise precision, recall and support for each class
	return prec, rec, f_meas, support
		
