import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 500 
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 #we have digits 0 to 9

batch_size = 100 #we'll be training the NN with 100 images at a time

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#setting up the computational graph

def neural_network_model(data):
	#input_data * weights + biases
	#biases are needed because if all the input data was zero, no neuron would ever fire

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])), 
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

	num_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size) #run through the data for us
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y}) #c represents cost
				epoch_loss += c
		
			print 'Epoch', epoch , 'completed out of', num_epochs, 'loss', epoch_loss
		
		#checking NN with the test data.Very high level
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
			
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})*100
train_neural_network(x)