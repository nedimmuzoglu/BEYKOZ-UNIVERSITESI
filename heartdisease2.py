# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 02:03:11 2023

@author: nedim
"""

# https://youtu.be/j2kfzYR_abI
"""
@author: Sreenivas Bhattiprolu
This code makes sense when you watch the accompanying video:
 https://youtu.be/j2kfzYR_abI   
#Dataset link:
https://cdn.scribbr.com/wp-content/uploads//2020/02/heart.data_.zip?_ga=2.217642335.893016210.1598387608-409916526.1598387608
#Heart disease
The effect that the independent variables biking and smoking 
have on the dependent variable heart disease 
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import numpy as np

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('heart_data.csv')
print(df.head())

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


# Build the network
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu')) 
model.add(Dense(1)) 
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())


history = model.fit(X_train, y_train ,verbose=1, epochs=1000, 
                    validation_data=(X_test, y_test))

# Predict



prediction_test = model.predict(X_test)    
print(y_test, prediction_test)
print("Mean sq. errror between y_test and predicted =", np.mean(prediction_test-y_test)**2)


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

##############################

# Print weights
for layer_depth, layer in enumerate(model.layers):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    print('___________ Layer', layer_depth, '__________')
    for toNeuronNum, bias in enumerate(biases):
        print(f'Bias to Layer{layer_depth+1}Neuron{toNeuronNum}: {bias}')
    
    for fromNeuronNum, wgt in enumerate(weights):
        for toNeuronNum, wgt2 in enumerate(wgt):
            print(f'Layer{layer_depth}, Neuron{fromNeuronNum} to Layer{layer_depth+1}, Neuron{toNeuronNum} = {wgt2}')
#######################################################################            
"""
As the weights change for each run let us use the following weights for our calculations
___________ Layer 0 __________
Bias to Layer1Neuron0: 4.4128947257995605
Bias to Layer1Neuron1: 4.5146260261535645
Layer0, Neuron0 to Layer1, Neuron0 = -0.08574138581752777
Layer0, Neuron0 to Layer1, Neuron1 = -0.059531815350055695
Layer0, Neuron1 to Layer1, Neuron0 = 0.1630137860774994
Layer0, Neuron1 to Layer1, Neuron1 = -0.015843335539102554
___________ Layer 1 __________
Bias to Layer2Neuron0: 2.504946708679199
Layer1, Neuron0 to Layer2, Neuron0 = 1.4296010732650757
Layer1, Neuron1 to Layer2, Neuron0 = 1.1467727422714233
"""
        
x0 = 65.1292
x1 = 2.21956

hidden0 = max(0, ((x0*-0.08574138)+(x1*0.163013786)+(4.4128947)))
hidden1 = max(0, ((x0*-0.0595318)+(x1*-0.0158433)+(4.514626)))

output = max(0, ((hidden0*1.4296)+(hidden1*1.14677)+(2.504947)))
print(output)



#Split data into train and test to verify accuracy after fitting the model. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# import numpy as np
# unique_elements, counts_elements = np.unique(y_train, return_counts=True)
# print(np.asarray((unique_elements, counts_elements)))
####################################################################
# Over-Complex network
# More complex than needed 

model = Sequential()
model.add(Dense(128, input_dim=30, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dropout(0.5))
 model.add(Dense(1)) 
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
model.add(Dense(16, input_dim=30, activation='relu')) 
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
