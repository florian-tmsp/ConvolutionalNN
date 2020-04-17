# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 21:43:08 2020

@author: Flo
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#training

# création du réseau de neurones vide
classifier = Sequential()

# Couche de convolution :

#filters = Dimensionalité de l'espace de sortie / nombre de features detector on conseille des puissance de deux
# taille de la feature detectors (matrice carré)
#kernel_size = taille du filtre de convolution
#☺ strides = déplacement du filtre sur les pixels 
#input_shape =taille de l'entrée l'image, dimension de l'espace d'arrivé (RGB=3 / Grayscale = 1)
# fonction redresseur truc de pimpins
classifier.add(Convolution2D(filters = 32, kernel_size = 3, strides = 1, input_shape = (64, 64, 3), activation = 'relu' ))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units=128, activation = 'relu'))

#utiliser la fonction softmax si plusieur catégories
classifier.add(Dense(units=1, activation= 'sigmoid'))

#compilation (choix d'algorithme du gradient )
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Tour de passpint pour augmenter artificiellement la base de donnée


    #transformation

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        #modification de l'echelle des pixel (sortie de l'intervalle 0-255)
        rescale=1./255,
        #transvection  
        shear_range=0.2,
        #zoom
        zoom_range=0.2,
        #pivotement des images
        horizontal_flip=True)

    #modification d'echelle 
test_datagen = ImageDataGenerator(rescale=1./255)

    #création des nouvelles images
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        #taille des images
        target_size=(64, 64),
        #@comme d'hab  taille du lot avant rétro processing 
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        #chemin données test
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

    #entrainement du modèle
classifier.fit_generator(
        training_set,
        #taille du data training set / batch_size
        steps_per_epoch=250,
        epochs=10,
        validation_data=test_set,
        #taille data test set / batch_size
        validation_steps=63)
