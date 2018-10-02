# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 17:14:29 2018

@author: wanghao
"""
from   numpy import *
import h5py
import tensorflow as tf
import os
import sys


Path_Train_x = "./mnist_train_data_x/"
Path_Train_y = "./mnist_train_data_y/"
Path_Test_x = "./mnist_test_data_x/"
Path_Test_y = "./mnist_test_data_y/"
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

# wite text file
print("Extracting Training Data...")
for i in range (len(x_train[:,1,1])):
    with open(Path_Train_x+"trainSample_" + str(i+1) +'.txt', "w") as txt_outputfile:
        for r in range(len(x_train[1,:,1])):
            for c in range(len(x_train[1,1,:])):
                data_x = x_train[i,r,c]
                txt_outputfile.write(str(data_x)+' ')
            txt_outputfile.write('\n')
    txt_outputfile.close()
    sys.stdout.write('\r'"[%-40s] %d%%" % ('='*int(i/(len(x_train[:,1,1]))*40), int(100*(i/(len(x_train[:,1,1]))))))
    sys.stdout.flush()
print("\n")

print("Extracting Training Groundtruth...")    
with open(Path_Train_y+"trainSample_GT.txt", "w") as txt_outputfile:
    for r in range (len(y_train[:])):
        data_y = y_train[r]
        txt_outputfile.write(str(data_y))
        txt_outputfile.write('\n')
        sys.stdout.write('\r'"[%-40s] %d%%" % ('='*int(i/(len(y_train[:]))*40), int(100*(i/(len(y_train[:]))))))
        sys.stdout.flush()
txt_outputfile.close()
print('\n')

print("Extracting Test Data...")
for i in range (len(x_test[:,1,1])):
    with open(Path_Test_x+"testSample_" + str(i+1) +'.txt', "w") as txt_outputfile:
        for r in range(len(x_test[1,:,1])):
            for c in range(len(x_test[1,1,:])):
                data_x = x_test[i,r,c]
                txt_outputfile.write(str(data_x)+' ')
            txt_outputfile.write('\n')
    txt_outputfile.close()
    sys.stdout.write('\r'"[%-40s] %d%%" % ('='*int(i/(len(x_test[:,1,1]))*40), int(100*(i/(len(x_test[:,1,1]))))))
    sys.stdout.flush()
print('\n')

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

print("Extracting Test Results...")
with open(Path_Test_y+"testSample_Results.txt", "w") as txt_outputfile:
    for r in range (len(y_test[:])):
        data_y = y_test[r]
        txt_outputfile.write(str(data_y))
        txt_outputfile.write('\n')
        sys.stdout.write('\r'"[%-40s] %d%%" % ('='*int(i/(len(y_test[:]))*40), int(100*(i/(len(y_test[:]))))))
        sys.stdout.flush()
txt_outputfile.close()
