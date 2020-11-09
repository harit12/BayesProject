import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from glob import glob
import os
import spacy
import datetime
from collections import Counter
nlp = spacy.load('en_core_web_sm')
test = nlp("This is a test")
everyFile = '*.txt'
stop_words = nlp.Defaults.stop_words
print(datetime.datetime.now(), 'initial time')
class DataSet():

    def __init__(self, trainingdir):
        self.trainingDir = trainingdir
        self.data_raw_p, self.data_raw_n, self.data_raw = self.load_data()
        self.data_prep_p, self.data_prep_n = self.preProcessData(self.data_raw_p), self.preProcessData(self.data_raw_n)
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
                data.append(f.readlines())
        return data
    def data_labeler(self, data, label):
        """
        args: data(type: list, expl: review data), label(type: string, expl: the label wanted on data)
        Take data and label, and make dictionary where the key is one data value, and the value is the label
        return: (type: dictionary, expl: dictionary containing data and label)
        """
        #print("len(data) = {}".format(len(data)))
        #set_data = set(data)
        #print("len(set_data) = {}".format(len(set_data)))
        res = {review: label for review in data}
        #print("len(res) = {}".format(len(res)))
        #Debugging
        return res
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
    def flatten(self, input_list):
        """
        args: list(type: list, expl: 2D list to be flattened)
        Flatten a 2D list
        return: (type: list, expl: flattened list)
        """
        flattened = []
        for i in input_list:
            for x in i:
                flattened.append(x)
        return flattened
    def wordCounter(self, words):
        """
        args: words(type: list, expl: list of all words)
        Counts the words in list, and makes dictionary to store words and their respective counts. Gets rid of duplicate words by converting list to dictionary
        return: (type: dictionary, expl: dictionary of words and respective counts)
        """
        word_count = dict(Counter(words))
        return word_count
    def get_reviews(self, data):
        """
        args: data(type: dictionary, expl: dictionary to extract keys from)
        Get keys from dictionary and in this case, reviews from postive or negative review dictionary
        return: (type: list, expl: list containing keys)
        """
        res = list(data.keys())
        return res
    #API Functions
    def load_data(self):
        """
        args: none
        Gets directories of data preloaded with *.txt. Store data in list using data_loader function, and label data. Finally merge positive and negative data, and store into DataFrame
        return: negData(type: dictionary, expl: dictionary containing labeled negative reviews), posData(type: dictionary, expl: dictionary containing labeled positive reviews), data_mixed(type: DataFrame, expl: DataFrame containing review and classification)
        """
        posDir, negDir = self.preload()
        posData = self.data_loader(posDir)
        negData = self.data_loader(negDir)
        print(datetime.datetime.now(), 'time to load reviews from disk')
        posData = self.data_labeler(posData, 'pos')
        negData = self.data_labeler(negData, 'neg')
        data = self.merge_dict(posData, negData)
        columns = ['review', 'class']
        data_mixed = pd.DataFrame(list(data.items()), columns = columns)
        return negData, posData, data_mixed
    def preProcessReview(self, review):
        """
        args: review(type: String, expl: String containing the review)
        Tokenize all words and remove non-alphabetic characters and stop words
        return: (type: list, expl: list of words, preprocessed)
        """
        review_nlp = nlp(review)
        tokens = [token for token in review_nlp if token.is_alpha]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    def preProcessData(self, data):
        """
        args: data(type: dictionary, expl: the dictionary that contains labeled raw data of the reviews
        Get all the reviews from dictionary, and tokenize it. Remove all stop words, and then get word count of each word
        return: (type: dictionary, expl: dictionary containing words and their respective word counts)
        """
        reviewList = self.get_reviews(data)
        words = list(map(self.preProcessReview, reviewList))
        words = self.flatten(words)
        print(datetime.datetime.now(), 'time for loading in word list')
        res = self.wordCounter(words)
        print(datetime.datetime.now(), 'time at end')
        return res
#data = DataSet('aclImdb/train/')
class Vocab:
    def __init__(self,dataDir):
        self.vocab = self.extractVocab(dataDir)
        
    def extractVocab(self, dataDir):
        """
        args: dataDir(type: String, expl: the directory where vocab file is stored)
        Extract vocab from the vocab file
        return: (type: list, expl: list containing all vocab words)
        """
        vocab = []
        with open(os.path.join(dataDir, 'imdb.vocab'), 'r') as f:
            vocab = [lines.rstrip('\n') for lines in f.readlines()]
        return vocab
    
    def idx2word(self,index):
        vocab = self.vocab
        word  = vocab[index]
        return word
    def word2idx(self, word):
        vocab = self.vocab
        index = vocab.index(word)
        return index

vocabulary = Vocab('aclImdb')
print(vocabulary.idx2word(5))
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
        reviewInClass = [row['review'].lower() for index, row in train_data.iterrows() if row['class'] == classification]
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




    
