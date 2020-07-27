import os 
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow import feature_column
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
print(tf.__version__)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def unique(list1): 
  
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    for x in unique_list: 
        print(x)


plastML = pd.read_csv("/home/ahmad/scRNA-seqAnalysis/plastMLscopenames.csv")
plastML = plastML.drop("Unnamed: 0", axis=1)
plastML.rename(columns={'ct': 'target'}, inplace = True)
plastML.head()

plastML.shape
unique(plastML.target)

plastML.dtypes
plastML['target'] = pd.Categorical(plastML['target'])
plastML['target'] = plastML.target.cat.codes
plastML.head()

# for target: 
# 1 is endodermis
# 2 is hair cells
# 3 is meristem
# 4 is non hair cells
# 5 is phloem
# 6 is root cap cells
# 7 is stele
# 8 is xylem
# 9 is cortex

train, test = train_test_split(plastML, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

col = train.columns[1:1182,]

feature_columns = []
for header in col:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 500
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(300, activation= 'relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(50, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)