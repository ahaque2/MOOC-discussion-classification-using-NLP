#Basic class for classification
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

def classification(tfidf, label, comments, tag, sk_fold, algo, kernel):
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
	kf = model_selection.StratifiedKFold(n_splits = sk_fold, random_state=0, shuffle = False)
	for train_index, test_index in kf.split(tfidf,label):
		
		x_train, x_test = tfidf[train_index], tfidf[test_index]
		y_train, y_test = label[train_index], label[test_index]
		
		clf_list = []
		predict_list = []
		results = []
		
		if(algo=='NN'):
			scaler = StandardScaler()
			scaler.fit(x_train)  
			x_train = scaler.transform(x_train)  
			x_test = scaler.transform(x_test) 
			
		clf_list, predict_list, score, tag_clf = train_model_2(x_train, y_train, algo, kernel)
		models.append(clf_list)
		#joblib.dump(clf_list, str(i)+'_model_stemmed_Lem_3_20_iter.pkl')
		predictions.append(predict_list)
		test_x.append(x_test)
		test_y.append(y_test)
		testing_index.append(test_index)
		
		results, tag_probability = test_model_v3(x_test, y_test, clf_list, predict_list)
		acc, pr, r, f, confusion_matrix, class_report = compute_metric(results, test_index, label, tag_clf)
		prec, rec, f_meas, support = class_report_(class_report, tag)
		total_precision.append(pr)
		total_recall.append(r)
		accuracies.append(acc)
		total_f_measure.append(f)
		
		precision.append(prec)
		recall.append(rec)
		f_measure.append(f_meas)
		
		print("Start --------- ", i)
		#print("Score = ",score)
		print("number of training data ", len(train_index), "  Number of test data instances", len(test_index))
		#write_to_csv(results, tag_probability, test_index,i)
		
		print("Suuport for each class - \t", support)
		print("precision - ", pr, "\t\t", prec)
		print("recall - ", r, "\t\t\t", rec)
		print("f1-measure - \t\t\t", f_meas)
		print("Accuracy - ", acc)
		#print("Confusion Matrix - \n" , confusion_matrix)
		print("Ends here -----------\n")
		
		i = i+1
	
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
		
	f= open("Results/Results_SVM_Kernels/Results_"+str(kernel)+"_"+str(sk_fold)+"_.txt","w+")
	f.write("precision - " + str(sum(total_precision)/sk_fold)+ "\t"+ str(avg_precision) +"\n")
	f.write("recall - " + str(sum(total_recall)/sk_fold) + "\t\t" + str(avg_recall)+"\n")
	f.write("f1-measure - " + str(sum(total_f_measure)/sk_fold) + "\t" + str(avg_f_measure)+"\n")
	f.write("Accuracy - " + str(sum(accuracies)/sk_fold)+"\n")
	#write_to_csv(results, test_index,i)
	
	print("\n\nFinal results for this case ----------")
	print("precision - ", sum(total_precision)/sk_fold, "\t", avg_precision)
	print("recall - ", sum(total_recall)/sk_fold, "\t", avg_recall)
	print("f1-measure - ", sum(total_f_measure)/sk_fold, "\t",  avg_f_measure)
	print("Accuracy - ", sum(accuracies)/sk_fold)
	print("\n")
		
def ml_(x, y, algo,k):
	# x is vector, y is label, algo is the algorithm to be used
	#Default algo is SVM
	clf = svm.SVC(kernel=k,probability=True)
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
	elif(algo == 'ensemble'):
		clf1 = KNeighborsClassifier(n_neighbors=35)
		clf2 = svm.SVC(kernel='linear',probability=True)
		clf3 = RandomForestClassifier(n_estimators=10)
		clf = VotingClassifier(estimators=[('knn', clf1), ('svc', clf2), ('rf', clf3)], voting='soft', weights=[2,3,2])
		
	clf.fit(x, y)
	tag_clf = clf.classes_
	predict = clf.predict(x)
	score = cross_val_score(clf, x, y)
	#print("score = ", score)
	return clf, predict, score, tag_clf

def train_model_2(data, y, algo, kernel):
	
	clf_list = []
	predict_list = []
	label_list = []
	print('Train model')
	label = y.tolist()
	clf, predict, score, tag_clf = ml_(data, label, algo, kernel)
	clf_list.append(clf)
	predict_list.append(predict)
	
	return clf_list, predict_list, score, tag_clf


def test_model_v3(data, y, clf_list, predict_list):
	#list = ['S','P','O','B','T','C','Q','BQ','CQ']
	predict_list_results = []
	predict_probability_list = []
	#print('Test model')
	
	for clf in clf_list:
		predict = clf.predict(data)
		predict_probability = clf.predict_proba(data)
		predict_probability_list.append(predict_probability)
		predict_list_results = predict.tolist()
	
	return predict_list_results, predict_probability_list
	#return precision, recall

def compute_metric(results, test_index, label, tag):
	actual_label = []
	predicted_label = []
	for predicted, i in zip(results, test_index):	
		actual_label.append(label[i])
		predicted_label.append(predicted)
	#print("Actual = ", actual_label)
	#print("Predicted = ", predicted_label)
	accuracy = round(accuracy_score(actual_label,predicted_label),2)
	precision = round(precision_score(actual_label,predicted_label,average='weighted'),2)
	recall = round(recall_score(actual_label,predicted_label, average='weighted'),2)
	f_measure = round(f1_score(actual_label,predicted_label, average='weighted'),2)
	#print(tag)
	print("recall - ",round(recall_score(actual_label,predicted_label, average='macro'),2))
	print("precision - ", round(precision_score(actual_label,predicted_label,average='macro'),2))
	class_report = classification_report(actual_label, predicted_label, target_names=tag)
	print(class_report)
	classif_report = classifaction_report_csv(class_report)
	confusion_mat = confusion_matrix(actual_label, predicted_label)
	return accuracy, precision, recall, f_measure, confusion_mat, classif_report
	
def classifaction_report_csv(report):
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
    #dataframe.to_csv('classification_report.csv', index = False)
	
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
		
	return prec, rec, f_meas, support
		