#Naive Bayes
###How to use
####How to use DataSet and Vocab data types
#####DataSet:
The DataSet datatype preprocesses data to seperate data into positive and negative reviews, tokenizes and removes stop words, and also adds word counts to the data. The word counts can be accessed
in dictionary or bag of words format. The labeled raw data can be viewed as a dictionary or Pandas DataFrame.
######How to use:
To use the class, create an object using the constructer, which just takes in the the directory of the training data
#####Vocab
The Vocab dataset just extracts the vocab from the vocab file, if given, in a directory. It is quicker than the DataSet class, but does have stop words.
######How to use:
To use the Vocab class, create a Vocab object using the constructer, which takes in the directory where the vocab file is located
#####WARNING:
DATASET CLASS SHOULD ONLY BE USED IF THERE IS MORE TIME AS IT TAKES LONGER TO RUN

