import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 10 #we have digits 0 to 9

batch_size = 128 #we'll be training the NN with 100 images at a time

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#setting up the computational graph

def conv2d(x, W):
	return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(x):
	#                        size of window     movement of window
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME') 


def convolutional_neural_network(x):
	#input_data * weights + biases
	#biases are needed because if all the input data was zero, no neuron would ever fire
	#5x5 convolution, 1 input, 32 features/outputs
	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
			   'out':tf.Variable(tf.random_normal([1024,n_classes])),
	                  }

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			  'b_conv2':tf.Variable(tf.random_normal([64])),
			  'b_fc':tf.Variable(tf.random_normal([1024])),
			  'out':tf.Variable(tf.random_normal([n_classes])),
	                  }

	x = tf.reshape(x, shape = [-1,28,28,1])
	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
	conv1 = maxpool2d(conv1)
	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
	conv2 = maxpool2d(conv2)


	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	output = tf.matmul(fc, weights['out'])+ biases['out']




	return output

def train_neural_network(x):
	prediction = convolutional_neural_network(x)
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