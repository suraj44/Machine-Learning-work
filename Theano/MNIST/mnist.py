import theano
import theano.tensor as T
import numpy as np
import time
import gzip
import os
import sys
#import dill
import six.moves.cPickle as pickle

#theano.gpuarray.use("cuda0")



class hiddenlayer(object):
	def __init__(self,rng ,input,nin,nout):
		self.input= input
		W = np.asarray(rng.uniform(low = -1, high = 1, size = (nin, nout)), dtype = theano.config.floatX)

		self.W = theano.shared(value = W,borrow = 'True')
		b = np.zeros((nout,), dtype = theano.config.floatX)
		self.b = theano.shared(value =b, borrow = 'True')
		output  = T.dot(input, self.W) + self.b
		self.output = T.nnet.softmax(output)
		self.params = [self.W,self.b]




#Class for setting up weights and biases as well as the cost function. The errors() function is used while training the neural net
class logisticreg(object):

	def __init__(self,input,nin,nout):
		self.W = theano.shared(value = np.zeros((nin,nout), dtype = theano.config.floatX),borrow = True)
		self.b= theano.shared(value= np.zeros((nout,), dtype = theano.config.floatX ),borrow = True)

		self.output = T.nnet.softmax(T.dot(input,self.W) + self.b)
		self.prediction = T.argmax(self.output, axis =1)
		self.input = input
		self.params = [self.W,self.b]

	def cost(self,y):
		return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])
		#return T.mean((self.output - y)**2)
		#eturn T.nnet.binary_crossentropy(self.output,y).eval()
		#return T.mean(-(y.T*T.log(self.output) + (1-y).T*T.log(1-self.output)))
	def errors(self,y):
		return T.mean(T.neq(self.prediction,y))



#Class defining the structure od the neural net
class NeuralNet(object):
	def __init__(self,rng,input,nin,nhidden,nout):
		self.hiddenlayer = hiddenlayer(rng = rng, input = input, nin = nin, nout = nhidden)
		self.logreg = logisticreg(input = self.hiddenlayer.output,nin = nhidden, nout = nout)
		
		self.L = (self.hiddenlayer.W ** 2).sum()+ (self.logreg.W ** 2).sum() #regularization term
		self.L1 = (
            abs(self.hiddenlayer.W).sum()
            + abs(self.logreg.W).sum())
		self.cost = self.logreg.cost
		self.errors = self.logreg.errors
		self.params = self.hiddenlayer.params + self.logreg.params
		self.input = input





#function to the load the MNIST dataset and split it into : training set, validation set and test set.
def load_data(dataset):
	# Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

	def shared_dataset(data_xy, borrow=True):

	        data_x, data_y = data_xy
	        shared_x = theano.shared(np.asarray(data_x,
	                                               dtype=theano.config.floatX),
	                                 borrow=borrow)
	        shared_y = theano.shared(np.asarray(data_y,
	                                               dtype=theano.config.floatX),
	                                 borrow=borrow)

	        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


dataset = 'mnist.pkl.gz'
datasets = load_data(dataset)


train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
rng = np.random.RandomState(1234)



X = train_set_x
y = train_set_y
index = 0
#########################################################################################
def opt(): #optimises weights and biases by minimizing the cost function

	learning_rate = 0.1
	#print type(np.asarray(y[:10]).tolist())

	clf = NeuralNet(rng=rng, input = X, nin = 28 * 28,nhidden = 1568, nout = 10) #Hidden layer unit numbers with good results : #1568 #35

	cost = clf.cost(y) + 0.0005*clf.L #+ #0.00001*clf.L1

	batchsize =5000
	i = 0


	test = theano.function(inputs = [],outputs = [clf.errors(y)], givens = {X : test_set_x[i : i + 100], y : test_set_y[i : i + 100]})

	validate = theano.function(inputs= [], outputs = clf.errors(y), givens = {X : valid_set_x[i : i + 100], y : valid_set_y[i : i + 100]})
	gparams = [T.grad(cost, p) for p in clf.params]

	temp = 0
	updates = [(param, param - (learning_rate-temp) * gparam) for param, gparam in zip(clf.params, gparams)]


	#updates = [(clf.W, clf.W - LEARNING_RATE * Wgrad), (clf.b, clf.b -LEARNING_RATE*bgrad)]

	#test = theano.function(inputs = [], outputs = [output])#,on_unused_input='ignore' )
	train = theano.function(inputs = [],outputs = cost, updates =  updates, givens = {X : train_set_x[i: i + batchsize], y : train_set_y[i: i + batchsize]})

	finaltest = theano.function(inputs= [], outputs = [clf.errors(y)], givens = {X : test_set_x, y : test_set_y})


	bestloss = np.inf
	test_score= 0.
	validationfreq = 1000
	n = 0
	
	while True:
		n+=1
		#learning_rate -= 0.0001
		avg_cost = train()
		if n%validationfreq ==0:
			print n
			temp=0.01
			#batchsize = 2000
			#print  clf.errors(y).eval()
			validloss = validate()

			currentloss = np.mean(validloss)

			print "validation error", currentloss * 100
			if currentloss < bestloss:
				bestloss = currentloss
				testloss = test()
				testscore = np.mean(testloss)
				print testscore
				print "learing rate", learning_rate
				#tee = dill.dumps(clf)
				#with open('best_model.pkl','wb') as f:
				 #	dill.dump(clf,f)
				 	#tmp = [clf.logreg.W, clf.logreg.b,clf.hiddenlayer.W, clf.hiddenlayer.b,clf.cost,clf.params,clf.errors,clf.input,clf.logreg.output, clf.logreg.prediction]
				 	#pickle.dump(tmp,f)
			if n==9000:
				print 
				print 
				print
				print "DONE!"
				print "Running final test..."
				print "Test score after training the neural net:"
				print str(100 - np.mean(finaltest())*100) + "/100"
				break 

			#if testscore * 100 < 5:
			#	break


		i += 100
		if i > 50000:
			i = 0




while True:
	tmp = int(raw_input("1 to optimise, 2 to break: "))
	if tmp==1:
		opt()

	if tmp==2:
		break
