# The Iris Dataset
"""The Iris Data Set consists of 50 samples from each of three species of Iris
(Iris setosa, Iris virginica and Iris versicolor).

Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
For a reference, see the following papers:

R. A. Fisher. "The use of multiple measurements in taxonomic problems". Annals of Eugenics. 7 (2): 179–188, 1936.

Goal: construct a neural network that classifies each sample into the correct class, as well as applying
validation and regularization techniques."""

from numpy.random import seed

seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import seaborn as sns
from keras.callbacks import ModelCheckpoint


# feature_names is a key with the name of all 4 features.
def read_in_and_split_data(data):
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['irisType'])
    y.head()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=.1)
    return X, y, train_X, test_X, train_y, test_y


iris_data = datasets.load_iris()
print(iris_data.keys())
# iris is a dictionary. We can see it’s keys using
print(iris_data.target_names, iris_data.target)

X, y, train_X, test_X, train_y, test_y = read_in_and_split_data(iris_data)

# convert iris data into a data frame so that I can do some preliminary visual analysis
iris_df = pd.DataFrame(np.concatenate((iris_data.data, np.array([iris_data.target]).T), axis=1),
                       columns=iris_data.feature_names + ['target'])
print(iris_df.head())
# Bivariate Pairwise relationships between columns with seaborn library
g = sns.pairplot(iris_df, hue="target", height=3, diag_kind="kde")

print('\n X shape= {}'.format(X.shape),
      '\n y shape= {}'.format(y.shape),
      '\n train_X shape= {}'.format(train_X.shape),
      '\n train_y shape= {}'.format(train_y.shape),
      '\n test_X shape= {}'.format(test_X.shape),
      '\n test_y shape= {}'.format(test_y.shape))

print(X.head())
print(y.value_counts())
# Convert targets to a one-hot encoding (so that the y values are categorical
# and higher values wont be given a higher priority

train_y = tf.keras.utils.to_categorical(np.array(train_y))
test_y = tf.keras.utils.to_categorical(np.array(test_y))

# Build the neural network model

print(train_X.shape[1])  # =4 =number of features


# we don't care about the number of training samples in the first layer which is train_X.shape[0]

# The densely connected layer means that all nodes of previous layers are connected to all nodes of the current layers
def get_model(shape):
    model = Sequential([
        Dense(units=64, kernel_initializer=initializers.he_uniform(),
              bias_initializer=initializers.Ones(), activation='relu', input_shape=(shape,)),
        Dense(128, activation='relu'),
        # we could pass any activation such as sigmoid/linear/tanh but it is proved that relu performs the best in
        # these kinds of models
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
        # since it is a multi-class classification, if it was binary, we would use sigmoid instead
    ])
    return model


model = get_model(train_X.shape[1])


# logistic loss or multinomial logistic loss are other names for cross-entropy loss
def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


compile_model(model)



history = model.fit(train_X, train_y, epochs=500, validation_split=0.15, verbose=0)

# To prevent overfitting, lets regularize the model.
# Use the same specs as the original model, but add 2 dropout layers, weight decay, and batch normalisation

def get_reg_model(shape, rate, wd):
    reg_model = Sequential([
        Dense(units=64, kernel_regularizer=regularizers.l2(wd), kernel_initializer=initializers.he_uniform(),
              bias_initializer=initializers.Ones(), activation='relu', input_shape=(shape,)),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        # we could pass any activation such as sigmoid/linear/tanh but it is proved that relu performs the best in
        # these kinds of models
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        BatchNormalization(),
        Dense(64, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dense(64, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dropout(rate),
        Dense(64, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dense(64, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        Dense(3, activation='softmax')
        # since it is a multi-class classification, if it was binary, we would use sigmoid instead
    ])
    return reg_model


# Instantiate the model, using a dropout rate of 0.3 and weight decay coefficient of 0.001
reg_model = get_reg_model(train_X.shape[1], 0.3, 0.001)

compile_model(reg_model)

reg_history = reg_model.fit(train_X, train_y, epochs=500, validation_split=0.15, verbose=0)

# Regularization helped reduce overfitting the network

# Incorporate callbacks with early stopping and learning rate reduction on plateau
# Define the learning rate reduction function


def get_callbacks():
    # A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing,
    # considering the min_delta and patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=30, verbose=1)
    # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback
    # monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=20)
    # Model checkpoint =mc will save the best model observed during training as defined by a chosen performance
    # measure on the validation dataset.
    mc = ModelCheckpoint('call_best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    return early_stopping, learning_rate_reduction, mc


# Learning rate is a hyperparamenter that controls how quickly the model is adapted to the problem.
# As the neural network calculates the weights that are used to make the most accurate predictions, the amount
# that you change the model during each step in the search process (step size) is called the "learning rate".
# Generally, a large learning rate allows the model to learn faster, but may not have as good as predictions than using
# a smaller learning rate. However a smaller learning rate takes longer to train.


early_stopping, learning_rate_reduction, mc = get_callbacks()

call_history = reg_model.fit(train_X, train_y, epochs=500, validation_split=0.15,
                              callbacks=[early_stopping, learning_rate_reduction, mc], verbose=0)

# plot the accuracy and loss graphs One can use plt.subplots() to make all their subplots at once and it returns the
# figure and axes (plural of axis) of the subplots as a tuple. A figure can be understood as a canvas where you paint
# your sketch. Whereas, you can use plt.subplot() if you want to add the subplots separately. It returns only the
# axis of one subplot. However, plt.subplots() is preferred because it gives you easier options to directly customize
# your whole figure

# Run this cell to plot the epoch vs accuracy graph

plt, ((axes11,axes12,axes13),
      (axes21,axes22,axes23))= plt.subplots(2, 3, sharey=True) # use sharey or sharex=True if you want to use the same axes for plots
axes11.set_ylim(0,3)
axes21.set_ylim(0,1.3)

axes11.plot(history.history['loss'])
axes11.plot(history.history['val_loss'])
axes11.set_title('Not Regularized')
axes12.plot(reg_history.history['loss'])
axes12.plot(reg_history.history['val_loss'])
axes12.set_title('Regularized')
axes13.plot(call_history.history['loss'])
axes13.plot(call_history.history['val_loss'])
axes13.set_title('Using Callbacks')
axes11.set_ylabel('Loss')
axes21.plot(history.history['accuracy'])
axes21.plot(history.history['val_accuracy'])
axes22.plot(reg_history.history['accuracy'])
axes22.plot(reg_history.history['val_accuracy'])
axes23.plot(call_history.history['accuracy'])
axes23.plot(call_history.history['val_accuracy'])
axes23.legend(['Training', 'Validation'], loc='lower right')
axes21.set_ylabel('Accuracy')
axes22.set_xlabel('Epochs')
plt.tight_layout()
plt.show()
plt.waitforbuttonpress()

# Evaluate the model on the test set
# Loss is the sum of errors (difference between predicted & actual value)).
# accuracy is used to measure the algorithm’s performance. It is a percentage #correct/total.
test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print("No Regularization")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(test_loss, 100 * test_acc))

reg_test_loss, reg_test_acc = reg_model.evaluate(test_X, test_y, verbose=0)
print("With Regularization")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(reg_test_loss, 100 * reg_test_acc))


saved_model = tf.keras.models.load_model('call_best_model.h5')
saved_test_loss, saved_test_acc = saved_model.evaluate(test_X, test_y, verbose=0)
print("Saved Reg. Model using Callbacks")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(saved_test_loss, 100 * saved_test_acc))

