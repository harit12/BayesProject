#Data Extraction
from glob import glob
import os
#AI/API
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
from collections import Counter
#Debugging
import datetime

nlp = spacy.load('en_core_web_sm')

#Debugging:
#test = nlp("This is a test")
#Static variables
everyFile = '*.txt'
stop_words = nlp.Defaults.stop_words

#Static methods
def flatten(input_list):
        """
        args: list(type: list, expl: 2D list to be flattened)
        Flatten a 2D list
        return: (type: list, expl: flattened list)
        """
        flattened = [word for sentence in input_list for word in sentence]
        return flattened

#Debugging:
#print(datetime.datetime.now(), 'initial time')

class DataSet():

    def __init__(self, trainingdir):
        self.trainingDir = trainingdir
        self.data_raw_n, self.data_raw_p, self.data_raw = self.load_data()
        self.data_prep_p = self.preProcessData(self.data_raw_p)
        self.data_prep_n =  self.preProcessData(self.data_raw_n)
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
        #Debugging:
        #print(negDir)
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
        args: (type: String, expl: Path of file(s))
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
    def get_reviews(self, data):
        """
        args: (type: dictionary, expl: dictionary to extract keys from)
        Get keys from dictionary and in this case, reviews from postive or negative review dictionary
        return: (type: list, expl: list containing keys)
        """
        res = list(data.keys())
        return res
    def load_data(self):
        """
        args: none
        Gets directories of data preloaded with *.txt. Store data in list using data_loader function, and label data. Finally merge positive and negative data, and store into DataFrame
        return: negData(type: dictionary, expl: dictionary containing labeled negative reviews), posData(type: dictionary, expl: dictionary containing labeled positive reviews), data_mixed(type: DataFrame, expl: DataFrame containing review and classification)
        """
        posDir, negDir = self.preload()
        posData = self.data_loader(posDir)
        negData = self.data_loader(negDir)
        #Debugging
        #print(datetime.datetime.now(), 'time to load reviews from disk')
        posData = self.data_labeler(posData, 'pos')
        negData = self.data_labeler(negData, 'neg')
        data = self.merge_dict(posData, negData)
        columns = ['review', 'class']
        data_mixed = pd.DataFrame(list(data.items()), columns = columns)
        return negData, posData, data_mixed
    def preProcessReview(self,review):
        """
        args: (type: String, expl: String containing the review)
        Tokenize all words, covert tokens to Strings, and remove non-alphabetic characters and stop words
        return: (type: list, expl: list of words, preprocessed)
        """
        review = review.lower()
        review_nlp = nlp(review)
        tokens = [str(token) for token in review_nlp if token.is_alpha]
        tokens = [token for token in tokens if token not in stop_words]
        return tokens
    def preProcessData(self, data):
        """
        args: (type: dictionary, expl: the dictionary that contains labeled raw data of the reviews
        Get all the reviews from dictionary, and tokenize it. Remove all stop words, and puncuation
        return: res(type: list, expl: 2D list containing tokens, and are all preprocessed)
        """
        reviewList = self.get_reviews(data)
        words = list(map(self.preProcessReview, reviewList))
        print(datetime.datetime.now(), 'time for loading in word list')
        return words
#data = DataSet('aclImdb/train/')
class Vocab:
    def __init__(self,vector):
        self.vocab = self.extractVocab(vector)
    #Helper Functions
    def extractVocab(self, vector):
        """
        args: (type: CountVectorizer, expl: bag of words vector)
        Extract vocab from the vocab file
        return: (type: list, expl: list containing all vocab words)
        """
        vocabList = vector.get_feature_names()
        return vocabList
    #API Functions
    def idx2word(self,index):
        """
        args: (type:
        """
        vocab = self.vocab
        word  = vocab[index]
        return word
    def word2idx(self, word):
        vocab = self.vocab
        index = vocab.index(word)
        return index
class Corpus:
    def __init__(self,word_list):
        tokenizer = lambda sentence: sentence.split()
        self.cv = CountVectorizer(tokenizer = tokenizer)
        self.word_list = word_list
        self.bow = self.create_bow()
        self.vocab = Vocab(self.cv)
        self.word_count_dict = self.create_word_count_dict()
    def create_bow(self):
        """
        args: None
        Creates bag of words vector using the CountVectorizer from sci-kit learn library
        return: (type: numpy matrix, expl: bag of words vector)
        """
        word_list = self.word_list
        corpData = [' '.join(sentence) for sentence in word_list]
        bow = self.cv.fit_transform(corpData)
        return bow
    def create_word_count_dict(self):
        """
        args: None
        Creates dictionary containing the word count of every word in the vocab list. 
        return: (type: dictionary, expl: dict containing every vocab word and its respective count)
        """
        bow = self.bow
        vocab_list = self.vocab.vocab
        count = np.sum(bow, axis = 0)
        count = flatten(count.tolist())
        word_count_dict = dict(zip(vocab_list, count))
        return word_count_dict
    #API functions
    def get_bow(self):
        bow = self.bow
        bow = bow.toarray()
        return bow
class Model:
    def __init__(self, trainingdir):
        self.data = DataSet(trainingdir)
        #Debugging:
        #print(self.data.data_prep_p)
        self.corpus_p = Corpus(self.data.data_prep_p)
        self.corpus_n = Corpus(self.data.data_prep_n)
        self.vocab_p = self.corpus_p.vocab
        self.vocab_n = self.corpus_n.vocab
        self.posProb = len(self.data.data_raw_p)/len(self.data.data_raw)
        self.negProb = len(self.data.data_raw_n)/len(self.data.data_raw)

    def probGivenWord(self, word, vocab, corpus, total_count, totalFeatures):
        """
        args: word(type: String, expl: Word inputed to find probablity of it given the class, vocab(type: Vocab, expl: Vocab object of the respective class), corpus(type: Corpus, expl: Corpus object containing bag of words vector and dictionary), total_count(type: int, expl: total ammount of times all features appear in data), totalFeatures(type: int, expl: total number of features)
        Checks if word is vocab list, and if it is, it assigns the count of the word to equal the amount of times it appears in the dataset. Uses laplace smoothing to find the probability of the word given the class
        return: (type: float, expl: probability of the word given the class using laplace smoothing)
        """
        if word in vocab.vocab:
            count = corpus.word_count_dict[word]
        else:
            count = 0
        laplaced = (count+1)/(total_count+totalFeatures)
        return laplaced

    def classifier(self, classification):
        """
        args: (type: String, expl: Checks class to see which bag of words to use)
        Checks the classification to either use the positive data or negative data
        return: vocab(type: Vocab, expl: Vocab object containing the vocabulary list), corpus(type: Corpus, expl: Corpus object containing the bag of words vector)
        """
        if classification == 'pos':
            vocab  = self.vocab_p
            corpus = self.corpus_p
        else:
            vocab = self.vocab_n
            corpus = self.corpus_n
        return vocab, corpus

    def probGivenClass(self, sentence, classification):
        """
        args: sentence(type: String, expl: Sentence wanted to find probablity of sentence given a class), classification(type: String, expl: class given to find probability of sentence given the class)
        Gets a sentence and classification and returns P(sentence|classification) or probability of sentence. This is done by finding the probability of a word given the class. Laplaced smoothing is apploed becuase some words may not be encountered in the training dataset
        return: (type: float, expl: P(sentence|classification) or probability of the sentence given the class)
        """
        data = self.data
        vocab, corpus = self.classifier(classification)
        totalFeatures = len(vocab.vocab)
        bow = corpus.bow.toarray()
        total_count = bow.sum()
        #DebuggingL
        #print(total_count, 'total')
        #print(totalFeatures, 'total Feat')
        sentence = data.preProcessReview(sentence)
        prob_list = [self.probGivenWord(word, vocab, corpus, total_count, totalFeatures) for word in sentence]
        prob = np.array(prob_list)
        final_prob  = np.prod(prob)
        #Debugging:
        #print(datetime.datetime.now(), 'time finish prob')
        return final_prob

    #API Functions
    def finalProb(self, review):
        """
        args: review(type: String, expl: review to classify
        Use probGivenClass function to find probablitlies of review given class, and use the probabilities of the each class to find the probablility of class given a review. Uses the naives bayes formula: P(class|sentence) = (P(class)P(sentence|class))/(P(otherclass))
        return: probPosGivenReview(type: float, expl: the probablity of the review being positive), probNegGivenReview(type: float, expl: the probabilty of review being negative), result(type: String, expl: String containing whether probability is negative or positive)
        """
        pPos = self.posProb
        pNeg = self.negProb
        probReviewGivenPos = self.probGivenClass(review, 'pos')
        probReviewGivenNeg = self.probGivenClass(review, 'neg')
        #Debugging:
        #print(probReviewGivenPos)
        probPosGivenReview = probReviewGivenPos*pPos/(probReviewGivenPos*pPos+probReviewGivenNeg*pNeg)
        probNegGivenReview = probReviewGivenNeg*pNeg/(probReviewGivenNeg*pNeg+probReviewGivenPos*pPos)
        result = ""
        if probPosGivenReview > probNegGivenReview:
            result = "positive"
        else:
            result = "negative"
        return probPosGivenReview, probNegGivenReview, result
def main():
    pass
if __name__ =='__main__':
    main()


    
