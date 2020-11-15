# Naive Bayes
### How to use
#### iPython
You can use the script via iPython by importing the Model class from the movieviewerModel module. From here, a Model object can be created via constructer, which inputs the training directory, which in this case is aclImdb/train. This will take about 14 minutes, so maybe get some food while the model trains. View the code if there are any errors 
Example:
```
[1] from movieviewerModel import Model

[2] model = Model('aclImdb/train')
```
*This step will take 14 minutes about*
After the model trains, the user can use different API methods. The main API method is finalProb method of the Model class. It takes a String input and returns if the probabilities of it being negative and positive. Another API method is the get_bow method of the Corpus class, which returns the bag of words vector. This can be accessed by getting the corpus instance variable inside of the model object. There are two corpus variables: corpus_n is for the negative training data, corpus_p is for the positive training data. The last API methods are word2idx and idx2word, which methods of the Vocab class. This can be accessed by using the 2 vocab instance variable, where vocab_p is for positive training data, and vocab_n is for negative training data. 
Examples:
```
[3] model.finalProb("This movie is horrible")
*output*
[4] model.corpus_p.get_bow()
*output*
[5] model.vocab.idx2word(4)
*output*
[6] model.vocab.word2idx('word');
*output*
```
#### Inside script
You can also edit the script, and change the main function to what you would like. You can use the API functions to in the main function and run movieviewerModel.py directly too.
