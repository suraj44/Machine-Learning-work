# Titanic

![Titanic Ship GIF](https://media.giphy.com/media/fxewISf5Z9F72/giphy.gif)

This is a simple feedforward and backpropogation artificial neural network written in TensorFlow to predict whether passengers on the Titanic Ship Tragedy are likely to have survived or not. There is extensive preprocessing done on the data using Sklearn's functions.

Here's the dataset before preprocessing:

![Original Dataset](https://i.imgur.com/Dlj4GRr.png)

And here's the dataset after preprocessing:

![Preprocessed Dataset](https://i.imgur.com/Pjqw4YN.png)


Running it for greater than 30 epochs makes it overfit and makes it more likely for it to predict that all passengers do not survive as that tends to reduce the cost function pretty well.

The dataset was obtained from Kaggle and it on 30 epochs, it achieves an accuracy of ~67% on the test data.

A .csv file is created that generates the prediction on the test data.

The training and testing data can be found in the Data sub-directory. 

Prerequisites to run:
1. Python 3
2. TensorFlow
3. Numpy




