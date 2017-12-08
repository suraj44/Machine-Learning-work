import tensorflow as tf
import pickle
import numpy as np
from create_sentiment_featuresets import create_feature_sets_and_labels
train_x, train_y, test_x, test_y = pickle.load(open("sentiment_set.pickle","rb"))

n_nodes_hl1 = 1500 
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500

n_classes = 2 
num_epochs = 10

batch_size = 100 #we'll be training the NN with 100 images at a time

x = tf.placeholder('float')
y = tf.placeholder('float')

#setting up the computational graph

def neural_network_model(data):
	#input_data * weights + biases
	#biases are needed because if all the input data was zero, no neuron would ever fire

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]),n_nodes_hl1])), 
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])), 
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])), 
	                  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])), 
	                  'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data , hidden_1_layer['weights']) , hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) #threshold function

	l2 = tf.add(tf.matmul(l1 , hidden_2_layer['weights']) , hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2 , hidden_3_layer['weights']) , hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3 , output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#now we want to minimise this cost

	optimizer = tf.train.AdamOptimizer().minimize(cost) #similar to stochastic gradient descent with learning rate of 0.0001

	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			i =0 
			while i<len(train_x):
				start = i 
				end = i + batch_size

				epoch_x = np.array(train_x[start:end])
				epoch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y}) #c represents cost
				epoch_loss += c
				i+= batch_size
		
			print 'Epoch', epoch+1 , 'completed out of', num_epochs, 'loss', epoch_loss
		
		#checking NN with the test data.Very high level
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
			
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Accuracy:', accuracy.eval({x:test_x, y:test_y})*100
train_neural_network(x)