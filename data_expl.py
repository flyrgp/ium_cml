import pandas as pd
import numpy as np

data = pd.read_csv('./googleplaystore.csv')

data.dropna(subset=['Rating', 'Type','Content Rating','Current Ver','Android Ver'], inplace=True)
data.reset_index(drop=True, inplace=True)
data = data.drop(columns=["Size", "Android Ver", "Current Ver", "Last Updated"])

# normalizing text
to_lowercase = ['App', 'Category', 'Type', 'Content Rating', 'Genres']
for column in to_lowercase:
    data[column] = data[column].apply(str.lower)

data["Installs"] = data["Installs"].replace({'\+': ''}, regex=True)
data["Installs"] = data["Installs"].replace({',': ''}, regex=True)
data["Price"] = data["Price"].replace({'\$': ''}, regex=True)

data["Genres"] = data["Genres"].astype('category')
data["Genres_numeric_value"] = (data["Genres"].cat.codes).astype('float32')

# normalizing numbers
data["Reviews"] = pd.to_numeric(data["Reviews"], errors='coerce')
max_value = data["Reviews"].max()
min_value = data["Reviews"].min()
data["Reviews"] = (data["Reviews"] - min_value) / (max_value - min_value)

data["Installs"] = pd.to_numeric(data["Installs"], errors='coerce')
max_value = data["Installs"].max()
min_value = data["Installs"].min()
data["Installs"] = (data["Installs"] - min_value) / (max_value - min_value)

data["Rating"] = np.asarray(data["Rating"]).astype('float32')
data["Reviews"] = np.asarray(data["Reviews"]).astype('float32')
data["Installs"] = np.asarray(data["Installs"]).astype('float32')
data["Price"] = np.asarray(data["Price"]).astype('float32')


print(data)


# splitting into sets
np.random.seed(123)
train, validate, test = np.split(data.sample(frac=1, random_state=42), [int(.6*len(data)), int(.8*len(data))])
print(f"Data shape: {data.shape}\nTrain shape: {train.shape}\nTest shape: {test.shape}\nValidation shape:{validate.shape}")

train.to_csv('apps_train.csv')
test.to_csv('apps_test.csv')
validate.to_csv('apps_validate.csv')
