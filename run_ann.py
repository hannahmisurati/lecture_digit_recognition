import keras
from keras.models import Sequential
from keras import layers

import pandas
import numpy


# Load the dataset from the CSV file
dataset = pandas.read_csv("dataset.csv")

# Shuffle the dataset
dataset = dataset.sample(frac=1)

# Separate target labels and input features
target = dataset.iloc[:,-1].values
data = dataset.iloc[:,:-1].values

# Normalize the input features to the range [0, 1]
data = data/255.0

# Initialize a sequential model
machine = Sequential()

# Add layers to the model
machine.add(layers.Dense(512, 
            activation="relu", 
            input_shape=(data.shape[1],)  
            ))
machine.add(layers.Dense(128, 
            activation="relu"))
machine.add(layers.Dense(64, 
            activation="relu"))
machine.add(layers.Dense(10, activation="softmax"))

# Compile the model
machine.compile(optimizer="sgd", 
                loss="sparse_categorical_crossentropy", 
                metrics=['accuracy'])
  

# Train the model on the dataset
machine.fit(data, target, epochs=30, batch_size=64)



# Load new data from the CSV file
new_data = pandas.read_csv("new_data.csv")

# Extract filenames from the new data
filename_list = new_data.iloc[:,-1].values

# Normalize the new data to the range [0, 1]
new_data = new_data.iloc[:,:-1].values
new_data = new_data/255.0

# Make predictions on the new data
prediction = numpy.argmax(machine.predict(new_data), axis=-1)

# Create a DataFrame to store the results
result = pandas.DataFrame()
result['filename'] = filename_list
result['prediction'] = prediction

# Print the result
print(result)


#The dataset is loaded from the CSV file and shuffled.
#Target labels and input features are separated.
#Input features are normalized to the range [0, 1].
#A sequential model is initialized.
#Dense layers are added to the model.
#The model is compiled with an optimizer, loss function, and evaluation metric.
#The model is trained on the dataset.
#New data is loaded from the CSV file.
#Filenames are extracted from the new data.
#New data is normalized.
#Predictions are made on the new data.
#Results are stored in a DataFrame and printed.























