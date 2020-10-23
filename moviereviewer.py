import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from glob import glob
##Import Datax
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
def probGivenClass(sentence,classification):
    """Find probabilies of a review given the classification"""
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
reviewTest = input()
##testFile = open(reviewTest, 'r').readlines()
wordProb  , probReviewGivenPos = probGivenClass(reviewTest, 'pos')
wordProbTwo2, probReviewGivenNeg = probGivenClass(reviewTest, 'neg')
probPosGivenReview = probReviewGivenPos*pPos/(probReviewGivenPos*pPos+probReviewGivenNeg*pNeg)
print('Probability of review being positive:',probPosGivenReview)
probNegGivenReview = probReviewGivenNeg*pNeg/(probReviewGivenNeg*pNeg+probReviewGivenPos*pPos)
print('Probability of review being negative:',probNegGivenReview)
if probPosGivenReview>probNegGivenReview:
    print("The review is positive")
else:
    print("The review is negative")
