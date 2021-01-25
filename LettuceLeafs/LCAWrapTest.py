from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import sys
from LCAWrapper import *



import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_covtype

# Helper libraries
import numpy as np
import pandas as pd


def load_data_covtype():
    print("Loading forest cover dataset...")

    cover = fetch_covtype()
    df = pd.DataFrame(cover['data'], columns=cover['feature_names'])
    target = cover['target']
    target=target-1
    target = to_categorical(target,7)

    return train_test_split(np.array(df)[:10000], np.array(target)[:10000],test_size=.3)

def load_fashion():
    print('TensorFlow version: {}'.format(tf.__version__))

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_labels = to_categorical(train_labels,10)
    test_labels = to_categorical(test_labels,10)
    # scale the values to 0.0 to 1.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # reshape for feeding into the model
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
    print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))
    return train_images,test_images,train_labels,test_labels,class_names



class MyTestCase(unittest.TestCase):
    def test_model_2d(self):
        XT,xt,YT,yt,classnames = load_fashion()
        print(XT.shape,YT.shape)
        input = keras.Input(shape=(XT.shape[1],XT.shape[2],1))
        X = keras.layers.Conv2D(5,5,activation='relu')(input)
        X = keras.layers.BatchNormalization()(X)
        X = keras.layers.Dropout(.25)(X)
        X = keras.layers.Conv2D(5,5,activation='relu')(X)
        X = keras.layers.GlobalMaxPool2D()(X)
        X = keras.layers.Dense(100)(X)
        out = keras.layers.Dense(YT.shape[1],activation='softmax')(X)
        lca_model = LCAWrap(inputs=input,outputs=out)
        lca_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        lca_model.Fit(x=XT,y=YT,validation_data=(xt,yt),epochs=2)
        lca_model.lca_out(path = '/home/kelly/PycharmProjects/Scratch/',name='2D.h5')
        print(lca_model.LCA_vals)

    def test_model_1d(self):
        XT,xt,YT,yt = load_data_covtype()
        print(XT.shape,xt.shape,YT.shape,yt.shape)
        input = keras.Input(shape=(XT.shape[1]))
        X = keras.layers.Dense(5)(input)
        X = keras.layers.Dense(5)(X)
        X = keras.layers.Dense(10)(X)
        out = keras.layers.Dense(YT.shape[1], activation='softmax')(X)
        lca_model = LCAWrap(inputs=[input], outputs=out)
        lca_model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        lca_model.Fit(x=XT,y=YT,validation_data=(xt,yt),epochs=2)
        lca_model.lca_out(path = '/home/kelly/PycharmProjects/Scratch/',name='1D.h5')
        # print(lca_model.last_LCA)
        print(lca_model.LCA_vals)


    # def test_model_fit(self):
    #
    # def test_model_lca_storage(self):
    #
    # def test_memory_check(self):
    #
    # def test_lca_saving(self):


if __name__ == '__main__':
    unittest.main()
