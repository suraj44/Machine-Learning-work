# MNIST

http://yann.lecun.com/exdb/mnist/

![MNIST](https://cdn-images-1.medium.com/max/581/1*zY1qFB9aFfZz66YxxoI2aw.gif)

It's a dataset of handwritten digits compiled mainly from American highschool students. Each digit is a 28x28 pixel image. There are 60,000 training images and 10,000 images for testing.

I've trained three different networks on the dataset:

1. Artificial Neural Network (Simple feedforward and backpropagation)
2. Recurrent Neural Network (LSTM cell)
3. Convolutional Neural Network (two convolutional layers and one fully connected layer)

Each neural network was given 10 epochs to train. 

Here are the final results:

1. ANN - 95.08%
3. RNN - 98.22%
2. CNN - 97.48%

