#This a simple feedforward, backpropagation neural network built usinig NumPy written from scratch
#I've ran against the digits dataset from the Scikit-learn dataset  and it lends itself to be easy to work with
import numpy as np
import random
#from tensorflow.examples.tutorials.mnist import input_data
#	mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10 #we have digits 0 to 9
n_nodes_hl1 = 500 
n_nodes_hl2 = 500
batch_size = 128 #we'll be training the NN with 100 images at a time

def sigmoid(x):
	return 1/(1.0 + np.exp(-x))

def dsigmoid(y):
	return y*(1.0 - y)

class neural_network_model(object):
	def __init__(self, input, hidden1, hidden2, output):
		self.input = input+1
		self.h1 = hidden1
		self.h2 = hidden2
		self.out= output

		self.a_in  = [1.0] * self.input
		self.a_in = np.array(self.a_in)
		self.a_h1  = [1.0] * self.h1
		self.a_h1 = np.array(self.a_h1)
		self.a_h2  = [1.0] * self.h2
		self.a_h2 = np.array(self.a_h2)
		self.a_out = [1.0] * self.out
		self.a_out = np.array(self.a_out)

		self.w_in = np.random.randn(self.input, self.h1)
		self.w_h1 = np.random.randn(self.h1, self.h2)
		self.w_h2 = np.random.randn(self.h2, self.out)

		self.c_in  = np.zeros((self.input, self.h1))
		self.c_h1 = np.zeros((self.h1, self.h2))
		self.c_h2 = np.zeros((self.h2, self.out))

	def feedforward(self, inputs):
		if len(inputs) != self.input -1:
			raise ValueError('Wrong no of inputs\n')

		for i in range(len(inputs)):
			self.a_in[i] = inputs[i]
		for j in range(self.h1):
			sum = 0.0
			for i in range(self.input):
				sum+= self.a_in[i]* self.w_in[i][j]
			self.a_h1[j] = sigmoid(sum)

		for j in range(self.h2):
			sum = 0.0
			for i in range(self.h1):
				sum+= self.a_h1[i]* self.w_h1[i][j]
			self.a_h2[j] = sigmoid(sum)

		
		for j in range(self.out):
			sum = 0.0
			for i in range(self.h2):
				sum+= self.a_h2[i]* self.w_h2[i][j]
			self.a_out[j] = sigmoid(sum)

		return self.a_out[:]

	def backprop(self, targets, learning_rate):
		if len(targets) != self.out:
			raise ValueError('Wrong number of outputs\n')

		output_deltas = [0.0] * self.out	

		for k in range(self.out):
			error = -(targets[k]- self.a_out[k])
			output_deltas[k] = dsigmoid(self.a_out[k]) *error

		h2_deltas = [0.0] * self.h2

		for j in range(self.h2):
			error = 0.0
			for k in range(self.out):
				error += output_deltas[k] * self.w_h2[j][k]
			h2_deltas[j] = dsigmoid(self.a_h2[j]) * error

		for j in range(self.h2):
			for k in range(self.out):
				change = output_deltas[k] * self.a_h2[j]
				self.w_h2[j][k] -= learning_rate * change + self.c_h2[j][k]
				self.c_h2[j][k] = change

		h1_deltas = [0.0] * self.h1

		for j in range(self.h1):
			error = 0.0
			for k in range(self.h2):
				error += h2_deltas[k] * self.w_h1[j][k]
			h1_deltas[j] = dsigmoid(self.a_h1[j]) * error

		for j in range(self.h1):
			for k in range(self.h2):
				change = h2_deltas[k] * self.a_h1[j]
				self.w_h1[j][k] -= learning_rate * change + self.c_h1[j][k]
				self.c_h1[j][k] = change

		for j in range(self.input):
			for k in range(self.h1):
				change = h1_deltas[k] * self.a_in[j]
				self.w_in[j][k] -= learning_rate * change + self.c_in[j][k]
				self.c_in[j][k] = change

		error = 0.0
		for k in range(len(targets)):
			error += 0.5*(targets[k]- self.a_out[k])**2

		return error

	def train(self, patterns, iter = 30, learning_rate = 0.01):
		for i in range(iter):
			error = 0.0
			random.shuffle(patterns)
			for  p in patterns:
				inp = p[0]
				out = p[1]
				
				self.feedforward(inp)
				
				error += self.backprop(out, learning_rate)
			if i%10==0:
				print "error %f\n" % error

	def predict(self, X):

		predictions = []
		for p in X:
			predictions.append(self.feedforward(p))
		return predictions


	def test(self, t):
		total = 0
		counter = 0
		for test in t:
			total +=1
			#print np.argmax(test[1])n, np.argmax(self.feedforward(test[0])) 
			if np.argmax(np.array(test[1]) == np.argmax(np.array(self.feedforward(test[0])))):
				counter +=1
		accuracy = float(counter)/total
		print "Accuracy: ", accuracy



def final():


	def load_data():
		data = np.loadtxt('Data/sklearn_digits.csv', delimiter = ',')
		#scaling the data
		y = data[:,0:10]

		data = data[:,10:]
		data -= data.min()
		data /= data.max()

		out = []

		for i in range(data.shape[0]):
			tmp = list((data[i,:].tolist(), y[i].tolist()))
			out.append(tmp)

		return out

	


	X = load_data()

	

	NN = neural_network_model(64, 100,100, 10)

	NN.train(X)

	NN.test(X)
if __name__ == '__main__':
	final()
