# The Iris dataset
"""In this assignment, you will use the Iris dataset. It consists of 50 samples from each of three species of Iris
(Iris setosa, Iris virginica and Iris versicolor).

Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
For a reference, see the following papers:

R. A. Fisher. "The use of multiple measurements in taxonomic problems". Annals of Eugenics. 7 (2): 179–188, 1936.
Your goal is to construct a neural network that classifies each sample into the correct class, as well as applying
validation and regularisation techniques."""
from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets, model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import seaborn as sns


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
print(iris_data.target_names)

X, y, train_X, test_X, train_y, test_y = read_in_and_split_data(iris_data)

#convert iris data into a data frame so that I can do some preliminary visual analysis
iris_df= pd.DataFrame(np.concatenate((iris_data.data, np.array([iris_data.target]).T), axis=1), columns=iris_data.feature_names + ['target'])
print(iris_df.head())
# Bivariate Pairwise relationships between columns with seaborn library
#sns.pairplot(iris_df, hue="target", size=3, diag_kind="kde")

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

#Build the neural network model

print(train_X.shape[1])  # =4 =number of features
#we don't care about the number of training samples in the first layer which is train_X.shape[0]

# The densely connected layer means that all nodes of previous layers are connected to all nodes of the current layers
def get_model(shape):
    model = Sequential([
        Dense(units=64, kernel_initializer=initializers.Identity(),
              bias_initializer=initializers.Ones(), activation='relu', input_shape=(shape,)),
        Dense(128, activation='relu'),
        # we could pass any activation such as sigmoid/linear/tanh but it is proved that relu performs the best in these kinds of models
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


def train_model(model, train_X, train_y, epochs, vs):
    history = model.fit(train_X, train_y, epochs=epochs, validation_split=vs, verbose=0)  # no output verbose=0
    return history

history = train_model(model, train_X, train_y, epochs=500, vs=.15)

test_loss, test_acc = model.evaluate(test_X, test_y, verbose=0)
print("Neural Net Model--No Regularization")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(test_loss, 100 * test_acc))

# To prevent overfitting, lets regularise the model.
# Use the same specs as the original model, but add 2 dropout layers, weight decay, and batch normalisation

def get_reg_model(shape, rate, wd):
    model = Sequential([
        Dense(units=64, kernel_regularizer=regularizers.l2(wd), kernel_initializer=initializers.Identity(),
              bias_initializer=initializers.Ones(), activation='relu', input_shape=(shape,)),
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation='relu'),
        # we could pass any activation such as sigmoid/linear/tanh but it is proved that relu performs the best in these kinds of models
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
    return model


# Instantiate the model, using a dropout rate of 0.3 and weight decay coefficient of 0.001
reg_model = get_reg_model(train_X.shape[1], 0.3, 0.001)

compile_model(reg_model)

reg_history = train_model(reg_model, train_X, train_y, epochs=500, vs=.15)
test_loss, test_acc = reg_model.evaluate(test_X, test_y, verbose=0)
print("Neural Net Model--With Regularization")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(test_loss, 100 * test_acc))

# Regularisation helped reduce overfitting the network

# Incorporate callbacks with early stopping and learning rate reduction on plateau
# Define the learning rate reduction function


def get_callbacks():
    # A model.fit() training loop will check at end of every epoch whether the loss is no longer decreasing,
    # considering the min_delta and patience
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=30)
    # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This callback
    # monitors a quantity and if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced.
    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=20)
    return early_stopping, learning_rate_reduction


call_model = get_reg_model(train_X.shape[1], 0.3, 0.0001)

compile_model(call_model)

early_stopping, learning_rate_reduction = get_callbacks()

call_history = call_model.fit(train_X, train_y, epochs=200, validation_split=0.15,
                              callbacks=[early_stopping, learning_rate_reduction], verbose=0)
print(call_history['early_stopping','learning_rate_reduction'])

# plot the accuracy and loss graphs

# Run this cell to plot the epoch vs accuracy graph
plt.clf()
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Not Regularised')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.subplot(1, 3, 2)
plt.plot(reg_history.history['accuracy'])
plt.plot(reg_history.history['val_accuracy'])
plt.title('Regularised')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.subplot(1, 3, 3)
plt.plot(call_history.history['accuracy'])
plt.plot(call_history.history['val_accuracy'])
plt.title('Using Callbacks')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()


# Run this cell to plot the epoch vs loss graph
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Not Regularised')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.subplot(1, 3, 2)
plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Reg Training', 'Reg Validation'], loc='upper right')
plt.subplot(1, 3, 3)
plt.plot(call_history.history['loss'])
plt.plot(call_history.history['val_loss'])
plt.title('Using Callbacks')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Reg Training', 'Reg Validation'], loc='upper right')
plt.show()

'''
try:
    plt.plot(call_history.history['accuracy'])
    plt.plot(call_history.history['val_accuracy'])
except KeyError:
    plt.plot(call_history.history['acc'])
    plt.plot(call_history.history['val_acc'])
plt.title('Using Callbacks: Accuracy vs. epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()
plt.plot(call_history.history['loss'])
plt.plot(call_history.history['val_loss'])
plt.title('Using Callbacks: Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()'''

# Evaluate the model on the test set
#Loss is the sum of errors (difference between predicted & actual value)).
# accuracy is used to measure the algorithm’s performance. It is a percentage #correct/total.

test_loss, test_acc = call_model.evaluate(test_X, test_y, verbose=0)
print("Neural Network Model with regularisation, callbacks, early stopping, and learning rate reduction on plateau")
print("Test loss: {:.3f}\nTest accuracy: {:.2f}%".format(test_loss, 100 * test_acc))
