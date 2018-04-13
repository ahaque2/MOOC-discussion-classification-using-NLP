Requirements (You need to have following to run this code in any environment) 

Python 3.0 or above (https://docs.python.org/3/installing/)
NLTK (https://www.nltk.org/install.html) -  NLTK is a leading platform for building Python programs to work with human language data
scikit-learn - (http://scikit-learn.org/stable/install.html)
Numpy - (https://docs.scipy.org/doc/numpy-1.14.0/user/install.html)

Running the code - 
The master file is "main_file.py" and is the only file that needs to be run (without any parameters). 

How the code works - 
main_file.py starts with initial Database "Data/comments.csv", and does some Data-preprocessing (Lemmatization, stemming, stop-word removal) and generates intermediate files and dumps in the folder "Data/". tf-idf is generated for the comments and stored in an np-array file under the same folder "Data/" as "tfidf.npy", along with other attribute np-array dumped under the same folder.
Once the Data-preprocessing is completed successfully, "main_file.py" triggers the training and subsequent testing of data on various classifiers specified in the list - 'algo'. I have used K-fold validation for validation of the results and metrics such as overall-accuracy, class precision, class recall, and class f1-score and a weighted precision value.

Results - 
The results include metrics such as overall-accuracy, class precision, class recall, and class f1-score and a weighted precision value for the task of classification for each classifier.

Classification categories for classification of comments - 
This code tries to process text from online student discussion forum and classifies them into one of the following categories:
(
O-discussion on subject theory, Q-Questions belonging to O category, 
B-Technical/software issue, BQ-Question belonging to B category, 
C-Logistics/deadline related discussion, 
S-Comments related to people socializing and introducing with each other, 
P- showing politeness, eg: saying thank you or appreciating something, 
T-Something that is completely off-topic and doesn't belong to any tag described earlier
)


Files/codes and it's function - 

main_file.py- The master file that needs to be run (without any params). This file calls all the other relevant code. 

Lematization.py - This does the data-preprocessing task of lemmatization using the Part-of-speech (POS) tagging. (Keep flag=1 if you want to remove proper nouns from the comments, by default it is set to 0), dumps the lemmatized comments into a file - 'Data/Lematized_comments.csv'.

stemming.py- This code does the preprocessing step of stemming the comments from Lemmatized comments using the data from - 'Data/Lematized_comments.csv'. After stemming the data is dumped into 'Data/stemmed_Lemetized_comments'.

tfidf.py - Computes the tf-idf of the for the comments from the file 'Data/stemmed_Lemetized_comments.csv' and creates an np-array and stores as tfidf.npy under the folder 'Data/'. Also creates np-array for other attributes like - thread_id, comment_positions, labels etc) and stores under the same 'Data/' folder for later use.

classification.py - This is the class that does the actual training and testing of each classifier. I have used K-fold validation and metrics like accuracy, precision, recall, and f1-score to represent my results.

thread_stats.py - Gives basic distribution of labels across comment_threads. Prints list of size equal to the number of threads and each value in the list corresponds to the number of such labels in that thread. Such list is produced for each label. A plot for each label is also plotted using this distribution.

class_distribution.py - Plots a pie chart of the distribution of each label across the dataset.


Directories - 

Data - Contains the initial cleaned data file (comments.csv). Also contains all the intermediate processed data files and the np-arrays to be used for training and classification.

Results - Contains Results of various runs. 1_Results, 2_Results, 3_Results correspond to results for 1-gram, 2-gram and 3-gram models. Results_only_Lem - contain results using only lemmatized comments. Results_only_stemming contains results for only stemmed comments. Results_removed_PN contains results for comments with Proper noun removed. PLots_for_Distributions contains plots for each tag across comment_thread.

