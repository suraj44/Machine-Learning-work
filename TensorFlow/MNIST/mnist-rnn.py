import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

num_epochs = 10
n_classes = 10 #we have digits 0 to 9
batch_size = 128 #we'll be training the NN with 100 images at a time
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

#setting up the computational graph

def recurrent_neural_network(x):
	#input_data * weights + biases
	#biases are needed because if all the input data was zero, no neuron would ever fire

	layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])), 
	                  'biases':tf.Variable(tf.random_normal([n_classes]))}
 	
 	x = tf.transpose(x, [1,0,2])
 	x = tf.reshape(x, [-1, chunk_size])
 	x = tf.split(0, n_chunks, x)
 	
 	lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)

 	outputs , states = rnn.rnn(lstm_cell, x,dtype = tf.float32)
	output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

	return output

def train_neural_network(x):
	prediction = recurrent_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	#now we want to minimise this cost

	optimizer = tf.train.AdamOptimizer().minimize(cost) #similar to stochastic gradient descent with learning rate of 0.0001

	

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x,epoch_y = mnist.train.next_batch(batch_size) #run through the data for us
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x.reshape((-1, n_chunks,chunk_size)), y:epoch_y}) #c represents cost
				epoch_loss += c
		
			print 'Epoch', epoch+1 , 'completed out of', num_epochs, 'loss', epoch_loss
		
		#checking NN with the test data.Very high level
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
			
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Accuracy:', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels})*100
train_neural_network(x)