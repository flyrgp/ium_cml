
import pandas as pd
import numpy as np
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# reading data
def read_data():
    all_data = []
    for name in ['train', 'test', 'validate']:
        all_data.append(pd.read_csv(f'data/apps_{name}.csv', header=0))
    return all_data

train_set, test_set, validate_set = read_data()
train_set = train_set.drop(columns=["Unnamed: 0"])
test_set = test_set.drop(columns=["Unnamed: 0"])
validate_set = validate_set.drop(columns=["Unnamed: 0"])
numeric_columns = ["Rating", "Reviews", "Installs", "Price", "Genres_numeric_value"]

# train set set-up
x_train_set = train_set[numeric_columns]
y_train_set = train_set["Category"]
encoder = LabelEncoder()
encoder.fit(y_train_set)
encoded_Y = encoder.transform(y_train_set)
dummy_y = np_utils.to_categorical(encoded_Y)

# validation set set-up
x_validate_set = validate_set[numeric_columns] 
y_validate_set = validate_set["Category"]
encoder = LabelEncoder()
encoder.fit(y_validate_set)
encoded_Yv = encoder.transform(y_validate_set)
dummy_yv = np_utils.to_categorical(encoded_Yv)

#test set set-up
x_test_set = test_set[numeric_columns]
y_test_set = test_set["Category"]
y_class_names = train_set["Category"].unique()
encoder = LabelEncoder()
encoder.fit(y_test_set)
encoded_Ytt = encoder.transform(y_test_set)
dummy_ytt = np_utils.to_categorical(encoded_Ytt)

try:
    no_epochs=int(sys.argv[1])
except:
    no_epochs = 200

# model definition
number_of_classes = 33
number_of_features = 5
model = Sequential()
model.add(Dense(number_of_classes, activation='relu'))
model.add(Dense(number_of_classes, activation='softmax',input_dim=number_of_features))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])
model.fit(x_train_set, dummy_y, epochs=no_epochs, validation_data=(x_validate_set, dummy_yv))
#model.save("my_model/")

#model predictions
#model = keras.models.load_model('my_model')
yhat = model.predict(x_test_set)
f = open("results.txt", "w")
for numerator, single_pred in enumerate(yhat):
    f.write(f"{sorted(y_class_names)[np.argmax(single_pred)]},{y_test_set[numerator]}\n")
f.close()
