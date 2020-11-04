import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from glob import glob
import os
import spacy

everyFile = '*.txt'
class DataSet():

    def __init__(self, trainingdir):
        self.trainingDir = trainingdir
        self.data_raw = self.load_data()
    # Helper/Util Function

    def data_splitter(self):
        """
        No args
        Split positive and negative data directories from training data
        return: (type: String, expl: Paths of positive and negative data directories)
        """
        posDir = ""
        negDir = ""
        for name in os.listdir(self.trainingDir):
            if name == 'pos':
                posDir = os.path.join(self.trainingDir, name)
            elif name == 'neg':
                negDir = os.path.join(self.trainingDir, name)
        print(negDir)
        return posDir, negDir

    def preload(self):
        """
        No args
        Get negative and positive directories and add *.txt to get everyfile text file in data
        return: (type: String, expl: pos and neg directories with *.txt at end of path)
        """
        posDir, negDir = self.data_splitter()
        posDir = os.path.join(posDir, everyFile)
        negDir = os.path.join(negDir, everyFile)
        return posDir, negDir

    def data_loader(self,dirFiles):
        """
        args: path of the files, string
        Use the path of the files to get all the data in them, and store the data in a list
        return: (type: list, expl: all data in the path in a list)
        """
        data = []
        for filename in glob(dirFiles):
            with open(filename, 'r') as f:
                data.append(f.readlines()[0])
        return data
    def data_labeler(self, data, label):
        """
        args: data(type: list, expl: review data), label(type: string, expl: the label wanted on data)
        Take data and label, and make dictionary where the key is one data value, and the value is the label
        return: (type: dictionary, expl: dictionary containing data and label)

        ISSUES: Loses some data
        """
        labelList = [label for review in data]
        labeled = dict()
        counter = 0
        for review in data:
            labeled[review] = label
        return labeled
    def merge_dict(self,dict1, dict2):
        """
        args: dict1(type: dictionary, expl: one of the two dictionaries), dict2(type: dictionary, expl: one of two dictionaries to be merged)
        Merge two dictionaries
        return: (type: dictionary, expl: merged dictionary)
        """
        dict3 = dict1.copy()
        for key, value in dict2.items():
            dict3[key] = value
        return dict3
    # API Functions

    def load_data(self):
        """
        args: none
        Gets directories of data preloaded with *.txt. Store data in list using data_loader function, and label data. Finally merge positive and negative data, and store into DataFrame
        return: (type: DataFrame, expl: DataFrame containing review and classification)

        ISSUE: Data loss due to data label function
        """
        posDir, negDir = self.preload()
        posData = self.data_loader(posDir)
        negData = self.data_loader(negDir)
        posData = self.data_labeler(posData, 'pos')
        negData = self.data_labeler(negData, 'neg')
        data = self.merge_dict(posData, negData)
        columns = ['review', 'class']
        return pd.DataFrame(list(data.items()), columns = columns)
    def preProcessData(self):
        """
        args: none
        Helper functions: tokenizer(tokenizes data) 
                          remove_stop_words(remove stop words and puncuations from data)
        return type is bag of words
        """
        return bag


data = DataSet('aclImdb/train/')
print(data.data)



class Model:
    # Import Data
    columns = ['review', 'class']
    data = [(open(filename, 'r').readlines()[0], 'pos')
            for filename in glob('aclImdb/train/pos/*.txt')]
    for filename in glob('aclImdb/train/neg/*.txt'):
        data.append((open(filename, 'r').readlines()[0], 'neg'))
    train_data = pd.DataFrame(data, columns=columns)
    # Find probabilite
    # Find probablilies of each classifications
    totalPosReviews = 0
    totalNegReviews = 0
    for index, row in train_data.iterrows():
        if row['class'] == 'pos':
            totalPosReviews += 1
        else:
            totalNegReviews += 1
    pPos = totalPosReviews/len(train_data)
    pNeg = totalNegReviews/len(train_data)

    def probGivenClass(self, sentence, classification):
        """Find probabilies of a review given the classification"""
        train_data = self.train_data
        reviewInClass = [row['review'].lower(
        ) for index, row in train_data.iterrows() if row['class'] == classification]
        # Get number of times word appears in sentence
        vectorizer = CountVectorizer()
        wordcounter = vectorizer.fit_transform(reviewInClass)
        count_list = wordcounter.toarray().sum(axis=0)
        freq_list = dict(zip(vectorizer.get_feature_names(), count_list))
        sentence = sentence.lower()
        sentence = sentence.split()
        allFeatures = [row['review'].lower()
                       for index, row in train_data.iterrows()]
        new_vec = CountVectorizer()
        new_vec.fit_transform(allFeatures)
        totalFeatures = len(new_vec.get_feature_names())
        total_count = count_list.sum()
        count = 0
        prob = []
        # Iterate through sentence, and check if any word match
        for word in sentence:
            if word in freq_list.keys():
                count = freq_list[word]
            else:
                count = 0
            prob.append((count+1)/(total_count+totalFeatures))
        prob_word = dict(zip(sentence, prob))
        num = np.product(prob)
        return prob_word, num

    def finalProb(self, review):
        pPos = self.pPos
        pNeg = self.pNeg
        wordProb, probReviewGivenPos = self.probGivenClass(review, 'pos')
        wordProbTwo2, probReviewGivenNeg = self.probGivenClass(review, 'neg')
        probPosGivenReview = probReviewGivenPos*pPos / \
            (probReviewGivenPos*pPos+probReviewGivenNeg*pNeg)
        probNegGivenReview = probReviewGivenNeg*pNeg / \
            (probReviewGivenNeg*pNeg+probReviewGivenPos*pPos)
        result = ""
        if probPosGivenReview > probNegGivenReview:
            result = "positive"
        else:
            result = "negative"
        return probPosGivenReview, probNegGivenReview, result
