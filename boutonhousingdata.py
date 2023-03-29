# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 02:14:37 2023

@author: nedim
"""

from keras.datasets import boston_housing
import numpy as np
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


max_target = np.max(train_targets)
     

train_data2 = train_data/np.max(train_data,axis=0)
test_data2 = test_data/np.max(train_data,axis=0)
train_targets = train_targets/max_target
test_targets = test_targets/max_target
     
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.regularizers import l1
model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam')
     


history = model.fit(train_data2, train_targets, validation_data=(test_data2, test_targets), epochs=100, batch_size=32, verbose=1)