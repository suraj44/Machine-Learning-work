# Sentiment Analysis using NLTK

I've used the NLTK (Natural Language Toolkit) module in Python to create a lexicon from a dataset of 'positive' and 'negative' sentences.

Here are some examples of 'positive' sentences from *pos.txt*:

must be seen to be believed . 
ray liotta and jason patric do some of their best work in their underwritten roles , but don't be fooled : nobody deserves any prizes here . 
everything that has to do with yvan and charlotte , and everything that has to do with yvan's rambunctious , jewish sister and her non-jew husband , feels funny and true . 
sweet home alabama " is what it is  a nice , harmless date film . . . 

Here are some examples of 'positive' sentences from *neg.txt*:

a puzzle whose pieces do not fit . some are fascinating and others are not , and in the end , it is almost a good movie . 
would that greengrass had gone a tad less for grit and a lot more for intelligibility . 
the good is very , very good . . . the rest runs from mildly unimpressive to despairingly awful . 

*create_sentimeent+featuresets.py* is for creating the lexicon of 423 words from the dataset. It is then pickled into sentiment_set.pickle (which has been zipped as it exceeds GitHub's 100MB file size limit)

*sentiment-neural-network* unpickles the lexicon and uses it in training an artificial neural network (simple feedforward and backpropagation)

A prediction accuracy of 59.38% was achieved.
