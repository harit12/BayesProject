import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from glob import glob
class Model:
    ##Import Data
    columns = ['review', 'class']
    data = [(open(filename, 'r').readlines()[0], 'pos') for filename in glob('aclImdb/train/pos/*.txt')]
    for filename in glob('aclImdb/train/neg/*.txt'):
        data.append((open(filename, 'r').readlines()[0], 'neg'))
    train_data = pd.DataFrame(data, columns=columns)
    ##Find probabilite
    ##Find probablilies of each classifications
    totalPosReviews = 0
    totalNegReviews = 0
    for index, row in train_data.iterrows():
        if row['class'] == 'pos':
            totalPosReviews+=1
        else:
            totalNegReviews+=1
    pPos = totalPosReviews/len(train_data)
    pNeg = totalNegReviews/len(train_data)
    def probGivenClass(self,sentence,classification):
       """Find probabilies of a review given the classification"""
       train_data = self.train_data
       reviewInClass = [row['review'].lower() for index, row in train_data.iterrows() if row['class']==classification]
       # Get number of times word appears in sentence
       vectorizer = CountVectorizer()
       wordcounter = vectorizer.fit_transform(reviewInClass)
       count_list = wordcounter.toarray().sum(axis=0)
       freq_list = dict(zip(vectorizer.get_feature_names(), count_list))
       sentence = sentence.lower()
       sentence = sentence.split()
       allFeatures = [row['review'].lower() for index, row in train_data.iterrows()]
       new_vec = CountVectorizer()
       new_vec.fit_transform(allFeatures)
       totalFeatures = len(new_vec.get_feature_names())
       total_count = count_list.sum()
       count = 0
       prob  = []
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
    def finalProb(self,review):
        pPos = self.pPos
        pNeg = self.pNeg
        wordProb  , probReviewGivenPos = self.probGivenClass(review, 'pos')
        wordProbTwo2, probReviewGivenNeg = self.probGivenClass(review, 'neg')
        probPosGivenReview = probReviewGivenPos*pPos/(probReviewGivenPos*pPos+probReviewGivenNeg*pNeg)
        probNegGivenReview = probReviewGivenNeg*pNeg/(probReviewGivenNeg*pNeg+probReviewGivenPos*pPos)
        result = ""
        if probPosGivenReview>probNegGivenReview:
            result = "positive"
        else:
            result = "negative"
        return probPosGivenReview, probNegGivenReview, result
       
class View:
    def initial(self):
        print("Type review:")
    def displayResults(self, pos, neg, res):
        print("The probability of review being positive is: ", pos)
        print("The probability of review being negative is: ", neg)
        print("Therefore, the review is ", res)
    
    

class Controller:
    def __init__(self,model, view):
       self.model = model
       self.view = view
    def sentenceGrabber(self):
        view = self.view
        view.initial()
        answer = input()
        return answer
    def getResults(self, answer):
        view = self.view
        model = self.model
        posProb, negProb, results = model.finalProb(answer)
        view.displayResults(posProb, negProb, results)
def main():
    model = Model()
    view = View()
    control = Controller(model, view)
    answer = control.sentenceGrabber()
    control.getResults(answer)
if __name__ == "__main__":
    main()
        
