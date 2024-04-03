import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy

from sklearn.model_selection import KFold
from sklearn import metrics

# Load the dataset from the CSV file
dataset = pandas.read_csv("dataset_extended.csv")

# Shuffle the dataset
dataset = dataset.sample(frac=1)

# Remove the last column (which is likely the label) from the dataset
dataset.drop(dataset.columns[-1], axis=1, inplace=True)

# Separate target labels and input features
target = dataset.iloc[:, -1].values
data = dataset.iloc[:, :-1].values

# Normalize the input features to the range [0, 1]
data = data / 255.0

# Reshape the input data to be compatible with Conv2D layers (expecting 28x28 images with 1 channel)
data = data.reshape(-1, 28, 28, 1)

# Define the number of splits for k-fold cross-validation
split_number = 4

# Initialize a k-fold object with the specified number of splits
kfold_object = KFold(n_splits=split_number)
kfold_object.get_n_splits(data)

# Lists to store accuracy and confusion matrices for each fold
results_accuracy = []
results_confusion_matrix = []

# Iterate over each fold in the k-fold cross-validation
for training_index, test_index in kfold_object.split(data):
    # Split the data into training and testing sets
    data_training = data[training_index]
    target_training = target[training_index]
    data_test = data[test_index]
    target_test = target[test_index]

    # Initialize a sequential model
    machine = Sequential()

    # Add Conv2D layers to the model
    machine.add(layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    machine.add(layers.Conv2D(64, kernel_size=(3, 3), activation="relu"))

    # Add MaxPooling2D layer to reduce spatial dimensions
    machine.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Add Dropout layer to prevent overfitting
    machine.add(layers.Dropout(0.25))

    # Flatten the feature maps for input to fully connected layers
    machine.add(layers.Flatten())

    # Add Dense layers to the model
    machine.add(layers.Dense(128, activation="relu"))
    machine.add(layers.Dropout(0.25))
    machine.add(layers.Dense(64, activation="relu"))
    machine.add(layers.Dense(10, activation="softmax"))

    # Compile the model
    machine.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    # Train the model on the training data
    machine.fit(data_training, target_training, epochs=30, batch_size=64)

    # Make predictions on the test data
    new_target = numpy.argmax(machine.predict(data_test), axis=-1)

    # Calculate and store accuracy for this fold
    results_accuracy.append(metrics.accuracy_score(target_test, new_target))

    # Calculate and store confusion matrix for this fold
    results_confusion_matrix.append(metrics.confusion_matrix(target_test, new_target))

# Print accuracy for each fold
print(results_accuracy)

# Print confusion matrix for each fold
for i in results_confusion_matrix:
    print(i)


#The dataset is loaded from the CSV file.
#The dataset is shuffled.
#The last column (which typically contains labels) is removed from the dataset.
#Target labels and input features are separated.
#Input features are normalized to the range [0, 1].
#Input data is reshaped to be compatible with Conv2D layers.
#KFold object is initialized for 4 splits.
#The model is trained and evaluated using k-fold cross-validation.
#Accuracy and confusion matrix are calculated and stored for each fold.
#Results are printed for each fold.
















