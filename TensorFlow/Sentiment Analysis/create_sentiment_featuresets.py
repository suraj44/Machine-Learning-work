'''We essentially have strings for different lengths that we need to feed to our neural network
We will be using the natural language processing module NLTK to lemmatize the words which would allow easier computation of the data'''

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
import random
import io
from collections import Counter

lemmatizer = WordNetLemmatizer()

num_lines = 100000 #just to make sure that we don't run out of RAM. Otherwise, we may get a MemoryError
#Function that will create a 'database' with all the words in our dataset, analogous to a vocabulary
def create_lexicon(pos, neg):
	
			lexicon = []
			for file in [pos,neg]:
				with io.open(file,'r', encoding = 'cp437') as f:
					contents = f.readlines()
					for line in contents[:num_lines]:
						all_words = word_tokenize(line.lower())
						lexicon += list(all_words)

			lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #stemming into legitimate words
			word_counts = Counter(lexicon) #returns a dictionary with each word as a key and number of occurences as its value
			l2 = [] #this will be our final lexicon
			for w in word_counts:
				if 1000>word_counts[w] > 50: #we're filtering the kind of words we want in our lexicon. Words like 'the', 'a' etc will have a very large number of occurences and we do not wish to include them
					l2.append(w)
			print(len(l2))
			return l2

def sample_handling(sample, lexicon, classification):
	featureset = []

	with io.open(sample, 'r', encoding = 'cp437') as f:
		contents = f.readlines()
		for line in contents[:num_lines]:
			current_words = word_tokenize(line.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])

	return featureset

def create_feature_sets_and_labels(pos,neg,test_size = 0.1):
	lexicon = create_lexicon(pos,neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1,0])
	features += sample_handling('neg.txt', lexicon, [0,1])
	random.shuffle(features)


	features = np.array(features)

	testing_size = int(test_size * len(features))

	train_x = list(features[:,0][:-testing_size]) #taking only the 0th indices
	#our featureset is of the form [[features, label]]. we wish to extract features

	train_y = list(features[:,1][:-testing_size]) #now we're taking the corresponding labels

	test_x = list(features[:,0][-testing_size:]) 

	test_y = list(features[:,1][-testing_size:]) 

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
