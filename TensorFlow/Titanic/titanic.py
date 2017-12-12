import pandas as pd
import tensorflow as tf
import numpy as np
import random
from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split



train_data = pd.read_csv(r"Data/train.csv")
test_data = pd.read_csv(r"Data/test.csv")
def NaN(data, columns):
	for column in columns:
		imputer = Imputer()
		data[column] = imputer.fit_transform(data[column].values.reshape(-1,1))
	return data


NaN_columns = ["Age", "SibSp", "Parch"]

train_data = NaN(train_data, NaN_columns)
test_data = NaN(test_data, NaN_columns)

test_passenger_id = test_data["PassengerId"]
#print(train_data)
def drop_not_concerned(data, columns):
	return data.drop(columns, axis =1)

not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]

train_data = drop_not_concerned(train_data, not_concerned_columns)
test_data = drop_not_concerned(test_data, not_concerned_columns)

def dummy_data(data, columns):
	for column in columns:
		data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
		data = data.drop(column, axis=1)
	return data

dummy_columns = ["Pclass"]

train_data=dummy_data(train_data, dummy_columns)
test_data=dummy_data(test_data, dummy_columns)

def sextoint(data):
	LE = LabelEncoder()
	LE.fit(["male", "female"])
	data["Sex"] = LE.transform(data["Sex"])
	return data

train_data = sextoint(train_data)
test_data = sextoint(test_data)

def normalize_age(data):
	scaler = MinMaxScaler()
	data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
	return data
train_data = normalize_age(train_data)
test_data = normalize_age(test_data)


def split_valid_test_data(data, fraction=(1 - 1)):
    data_y = data["Survived"]
    lb = LabelBinarizer()
    data_y = lb.fit_transform(data_y)

    data_x = data.drop(["Survived"], axis=1)

    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)

    return train_x.values, train_y, valid_x, valid_y
train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)
#print(train_x)
tmp = []
for i in train_y:
	if i==[0]:
		tmp.append([1,0])
	elif i==[1]:
		tmp.append([0,1])
train_y = tmp
tmp = []
for i in train_y:
	if i==[0]:
		tmp.append([1,0])
	elif i==[1]:
		tmp.append([0,1])
valid_y = tmp
tmp = []
for i in test_data:
	tmp.append([i])
print(tmp)

n_nodes_hl1 = 512
n_nodes_hl2 = 512
n_nodes_hl3 = 512

n_classes = 2 #we two classes: survived or not survived

batch_size = 10 #we'll be training the NN with 10 people at a time

x = tf.placeholder('float', [None, train_x.shape[1]])
y = tf.placeholder('float')
def neural_network_model(data):
	#input_data * weights + biases
	#biases are needed because if all the input data was zero, no neuron would ever fire

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([7,n_nodes_hl1])), 
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
	
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))

	optimizer = tf.train.AdamOptimizer().minimize(cost) #similar to stochastic gradient descent with learning rate of 0.0001

	num_epochs = 30

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
				#print(epoch_x)
				_, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y:epoch_y}) #c represents cost
				epoch_loss += c
				i+= batch_size
				
			print('Epoch', epoch+1 , 'completed out of', num_epochs, 'loss', epoch_loss)
			#print(y.shape())
		#checking NN with the test data.Very high level
		#predicting with test data
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			test_predict = sess.run(tf.argmax(prediction,1), feed_dict= {x:test_data})

		print(test_predict[:10])
		return test_predict

final =  train_neural_network(x)

tmp = []
#creating .csv file

passenger_id=test_passenger_id.copy()
evaluation=passenger_id.to_frame()
evaluation["Survived"]=final
print(evaluation[:10])

evaluation.to_csv("evaluation_submission.csv", index = False)




