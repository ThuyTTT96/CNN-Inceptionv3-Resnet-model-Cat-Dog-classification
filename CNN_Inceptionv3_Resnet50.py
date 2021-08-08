#import libraries
from numpy.core.fromnumeric import resize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
import os

from tensorflow.python.keras.backend import switch
from tqdm import tqdm 
from glob import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization , GlobalMaxPool2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50V2

# get image from path to folder 
def load_dataset(path):
    data_path = pathlib.Path(path)
    all_paths = data_path.glob('*/*.jpg') 
    all_paths = list(all_paths) 
    all_paths = list(map(lambda x: str(x), all_paths))
    return all_paths

def TestImageQuality(all_paths):
    new_all_paths = []
    for path in tqdm(all_paths):
        try :
            image = tf.io.read_file(path)
            image = tf.io.decode_jpeg(image , channels = 3)
        except :
            continue
        new_all_paths.append(path)
    return new_all_paths

def get_lable(image_path):
    return image_path.split('/')[-2]

#Use LabelEncoder when there are only two possible values of a categorical features.
def get_all_labels(all_paths):
    all_labels = list(map(lambda x: get_lable(x), all_paths))
    le = LabelEncoder()
    all_labels = le.fit_transform(all_labels)
    return all_labels

def load(image, label):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels = 3)
    return image, label

def get_dataset(paths, labels, train = True):
    image_paths    = tf.convert_to_tensor(paths)
    labels         = tf.convert_to_tensor(labels)

    image_dataset  = tf.data.Dataset.from_tensor_slices(image_paths)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)

    dataset        = tf.data.Dataset.zip((image_dataset, labels_dataset))

    dataset        = dataset.map(lambda image, label: load(image, label))
    dataset        = dataset.map(lambda image, label: ( resize(image),label), num_parallel_calls = AUTOTUNE)
    dataset        = dataset.shuffle(1000)
    dataset        = dataset.batch(batch_size)

    if train:
        dataset    = dataset.map(lambda image, label: (data_augementation(image), label), num_parallel_calls = AUTOTUNE)
        dataset    = dataset.repeat()
    return dataset

def build_model():
    model = Sequential()

    # Block 1 
    model.add(Conv2D(input_shape=(224 , 224 , 3),  padding='same',filters=32, kernel_size=(7, 7)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(filters=64,  padding='valid', kernel_size=(5, 5)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3 
    model.add(Conv2D(filters=128, padding='valid', kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    # Block 4 
    model.add(Conv2D(filters=256, padding='valid', kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=256 , kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(GlobalMaxPool2D())

    model.add(Dense(units=256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.add(Activation('sigmoid'))
    return model

def build_InceptionV3_model():
    backbone    = InceptionV3(
    input_shape = (224, 224, 3),
    include_top = False
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_Resnet50_model():
    backbone    = ResNet50V2(
    input_shape = (224, 224, 3),
    include_top = False
    )

    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def main():
    path = '/home/thuy/basicDL/CNN/PetImages'

if __name__=='__main__':
    path       = '/home/thuy/basicDL/CNN/PetImages'
    image_size = 224
    batch_size = 128
    AUTOTUNE   = tf.data.experimental.AUTOTUNE


    all_paths = load_dataset(path)
    shuffle(all_paths) #By shuffling,each data point creates an "independent" change on the model without being biased by the same points before them.
    ###
    all_paths  = TestImageQuality(all_paths)
    all_labels = get_all_labels(all_paths)
    
    #slip the data to trainning and testing dataset
    Train_paths, Val_paths, Train_labels, Val_labels = train_test_split(all_paths, all_labels)

    ##preprocessing data
    resize = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Resizing(image_size, image_size)])

    #Data Augmentation
    data_augementation = tf.compat.v1.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor = (-0.3, -0.2))
    ])
    

    ##get training and testing dataset
    train_dataset = get_dataset(Train_paths, Train_labels)
    val_dataset   = get_dataset(Val_paths, Val_labels, train = False)

    ## build model cnn
    #model = build_model()
    ## build inception v3
    #model = build_InceptionV3_model()
    ## build resnet50
    model = build_Resnet50_model()
    print(model.summary())


    ##complile and testing model
    model.compile(
            loss      = 'binary_crossentropy',
            optimizer = 'adam',
            metrics   = ['accuracy']
        )

    history = model.fit(
            train_dataset,
            steps_per_epoch  = len(Train_paths)//batch_size,
            epochs           = 10,
            validation_data  = val_dataset,
            validation_steps = len(Val_paths)//batch_size,
        )
    loss, acc = model.evaluate(val_dataset)
    print('Tess accuracy: ', acc)
    print('Test loss: ', loss)
    