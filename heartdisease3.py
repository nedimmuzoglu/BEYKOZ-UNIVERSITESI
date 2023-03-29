# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 02:10:38 2023

@author: nedim
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('heart_data.csv')

df = df.drop("Unnamed: 0", axis=1)
#A few plots in Seaborn to understand the data


sns.lmplot(x='biking', y='heart.disease', data=df)  
sns.lmplot(x='smoking', y='heart.disease', data=df)  


x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

x = x_df.to_numpy()
y = y_df.to_numpy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# import numpy as np
# unique_elements, counts_elements = np.unique(y_train, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))
####################################################################
# Over-Complex network
# More complex than needed 

# model = Sequential()
# model.add(Dense(128, input_dim=30, activation='relu')) 
# model.add(Dropout(0.5))
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(8))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1)) 
# model.add(Activation('sigmoid'))  
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',             #also try adam
#               metrics=['accuracy'])

# #model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())


##################################################################
# Complex network
# Still complex but may work...

# model = Sequential()
# model.add(Dense(16, input_dim=30, activation='relu')) 
# model.add(Dropout(0.5))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1)) 
# model.add(Activation('sigmoid'))  
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',             #also try adam
#               metrics=['accuracy'])

# #model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())
####################################################################
#Simple network 1
# Appropriate architecture for the challenge

# model = Sequential()
# model.add(Dense(16, input_dim=30, activation='relu')) 
# model.add(Dropout(0.2))
# model.add(Dense(1)) 
# model.add(Activation('sigmoid'))  
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',             #also try adam
#               metrics=['accuracy'])

# #model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())


#####################################################################

#Simple network 2
# Too simple??

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(1)) 
model.add(Activation('sigmoid'))  
model.compile(loss='binary_crossentropy',
              optimizer='adam',             #also try adam
              metrics=['accuracy'])

#model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
###########################################################

# Fit with early stopping and model checkpoint to save the best models. 
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# # patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
# mc = ModelCheckpoint('models/model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# # evaluate the model
# history = model.fit(X_train, y_train ,verbose=1, epochs=500, batch_size=64,
#                     validation_data=(X_test, y_test), callbacks=[es, mc])

#Fit with no early stopping or other callbacks
history = model.fit(X_train, y_train ,verbose=1, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test))

# Predict

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


# prediction_test = model.predict(X_test)    
# print(y_test, prediction_test)
# print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# from keras.models import load_model
# saved_model = load_model('models/best_model.h5')

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
