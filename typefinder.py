import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
columns = ['sentence', 'classification']
data = [['This is my book', 'stmt'], 
        ['They are novels', 'stmt'],
        ['have you read this book', 'question'],
        ['who is the author', 'question'],
        ['what are the characters', 'question'],
        ['This is how I bought the book', 'stmt'],
        ['I like fictions', 'stmt'],
        ['what is your favorite book', 'question']]
t_data = pd.DataFrame(data, columns=columns)
print(t_data)
def ProbGiven(sentence, classify):
    stmtAmmount = 0
    questionAmm = 0
    for data in t_data['classification']:
        if data == 'stmt':
            stmtAmmount+=1
        else:
            questionAmm+=1
    ##Find how many times the statement classification is in the dataset
    probStmt = stmtAmmount/(len(t_data.index))
    probQuestion = questionAmm/(len(t_data.index))
    ##Find how prob of sentence given class
    sentenceInClass = [row['sentence'] for index, row in t_data.iterrows() if row['classification']==classify]
    vectorizer = CountVectorizer()
    countMatrix = vectorizer.fit_transform(sentenceInClass)
    freq_words = dict(zip(vectorizer.get_feature_names(),countMatrix.toarray().sum(axis=0)))
    sentence = sentence.split()
    sentence = (word.lower() for word in sentence)
    data_sentences = [row['sentence'] for index, row in t_data.iterrows()]
    new_vectorizer = CountVectorizer()
    laplaceCount = new_vectorizer.fit_transform(data_sentences)
    totalFeatures = len(new_vectorizer.get_feature_names())
    print(totalFeatures)
    total_count = countMatrix.toarray().sum()
    print(total_count)
    count = 0
    laplaced = []
    for word in sentence:
        if word in freq_words.keys():
            count = freq_words[word]
        else:
            count = 0
        print(count)
        laplaced.append((count+1)/(total_count+totalFeatures))
    print(laplaced)
    print(np.prod(laplaced))
    if classify == 'stmt':
        numerator = np.prod(laplaced)*probStmt
    else:
        numerator =np.prod(laplaced)*probQuestion
    return numerator, classify
num, classify = ProbGiven("What is the price of the book", 'stmt')
print(num)
    
            

    
