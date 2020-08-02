#
# train.py
#
# Created by Samet Cetin.
# Contact: cetin.samet@outlook.com
#

import numpy as np
np.random.seed(123)
import gzip
import _pickle as cPickle
import os
import json
from collections import Counter

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KDTree

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical


WORD2VECPATH    = "../data/class_vectors.npy"
DATAPATH        = "../data/zeroshot_data.pkl"
MODELPATH       = "../model/"

def load_keras_model(model_path):
    with open(model_path +"model.json", 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path+"model.h5")
    return loaded_model

def save_keras_model(model, model_path):
    """save Keras model and its weights"""
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    model_json = model.to_json()
    with open(model_path + "model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_path + "model.h5")
    print("-> zsl model is saved.")
    return

def load_custom_data(dataset='awa'):
    train_data = None
    valid_data = None
    zero_shot_data = None
    test_data = None
    if dataset == 'awa':
        with open('../data/awa_train_data.json') as json_file:
            train_data = json.load(json_file)
        with open('../data/awa_validation_data.json') as json_file:
            valid_data = json.load(json_file)
        with open('../data/awa_zsl_data.json') as json_file:
            zero_shot_data = json.load(json_file)
        with open('../data/awa_test_data.json') as json_file:
            test_data = json.load(json_file)

    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    x_train, y_train    = zip(*train_data)
    y_train = np.squeeze(np.asarray(y_train))
    x_train = normalize(x_train, norm='l2')

    x_valid, y_valid = zip(*valid_data)
    y_valid = np.squeeze(np.asarray(y_valid))
    x_valid = normalize(x_valid, norm='l2')

    x_zsl, y_zsl = zip(*zero_shot_data)
    x_zsl = normalize(x_zsl, norm='l2')

    x_test, y_test = zip(*test_data)
    x_test = normalize(x_test, norm='l2')

    print("-> data loading is completed.")
    return (x_train, x_valid, x_zsl, x_test), (y_train, y_valid, y_zsl, y_test)


def load_data():
    """read data, create datasets"""
    # READ DATA
    with gzip.GzipFile(DATAPATH, 'rb') as infile:
        data = cPickle.load(infile)

    # ONE-HOT-ENCODE DATA
    label_encoder   = LabelEncoder()
    label_encoder.fit(train_classes)

    training_data = [instance for instance in data if instance[0] in train_classes]
    zero_shot_data = [instance for instance in data if instance[0] not in train_classes]
    # SHUFFLE TRAINING DATA
    np.random.shuffle(training_data)

    ### SPLIT DATA FOR TRAINING
    train_size  = 300
    train_data  = list()
    valid_data  = list()
    for class_label in train_classes:
        ct = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ct < train_size:
                    train_data.append(instance)
                    ct+=1
                    continue
                valid_data.append(instance)

    # SHUFFLE TRAINING AND VALIDATION DATA
    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)

    train_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15))for instance in train_data]
    valid_data = [(instance[1], to_categorical(label_encoder.transform([instance[0]]), num_classes=15)) for instance in valid_data]

    # FORM X_TRAIN AND Y_TRAIN
    x_train, y_train    = zip(*train_data)
    x_train, y_train    = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    # L2 NORMALIZE X_TRAIN
    x_train = normalize(x_train, norm='l2')

    # FORM X_VALID AND Y_VALID
    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    # L2 NORMALIZE X_VALID
    x_valid = normalize(x_valid, norm='l2')


    # FORM X_ZSL AND Y_ZSL
    y_zsl, x_zsl = zip(*zero_shot_data)
    x_zsl, y_zsl = np.squeeze(np.asarray(x_zsl)), np.squeeze(np.asarray(y_zsl))
    # L2 NORMALIZE X_ZSL
    x_zsl = normalize(x_zsl, norm='l2')

    print("-> data loading is completed.")
    return (x_train, x_valid, x_zsl), (y_train, y_valid, y_zsl)


def awa_kernel_init(shape):
    class_vectors = None
    with open('../data/vectors_data.json') as json_file:
            class_vectors = json.load(json_file)
    training_vectors    = sorted([(vector_data['label'], vector_data['vector']) for vector_data in class_vectors if vector_data['label'] in train_classes], key=lambda x: x[0])
    classnames, vectors = zip(*training_vectors)
    vectors             = np.asarray(vectors, dtype=np.float)
    vectors             = vectors.T
    return vectors

def custom_kernel_init(shape):
    class_vectors       = np.load(WORD2VECPATH,allow_pickle=True)
    training_vectors    = sorted([(label, vec) for (label, vec) in class_vectors if label in train_classes], key=lambda x: x[0])
    classnames, vectors = zip(*training_vectors)
    vectors             = np.asarray(vectors, dtype=np.float)
    vectors             = vectors.T
    return vectors

def  build_model(dataset='', is_fresh_model=True):
    model = None
    model = Sequential()
    model.add(Dense(1024, input_shape=(2048,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(NUM_ATTR, activation='relu'))
    if not is_fresh_model:
        model.load_weights(MODELPATH+"model.h5")
        print('loaded existing model')
    if dataset == 'awa':
        model.add(Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=awa_kernel_init))
    else:
        model.add(Dense(NUM_CLASS, activation='softmax', trainable=False, kernel_initializer=custom_kernel_init))

    print("-> model building is completed.")
    return model


def train_model(model, train_data, valid_data):
    x_train, y_train = train_data
    x_valid, y_valid = valid_data
    adam = Adam(lr=5e-5)
    model.compile(loss      = 'categorical_crossentropy',
                  optimizer = adam,
                  metrics   = ['categorical_accuracy', 'top_k_categorical_accuracy'])

    history = model.fit(x_train, y_train,
                        validation_data = (x_valid, y_valid),
                        verbose         = 2,
                        epochs          = EPOCH,
                        batch_size      = BATCH_SIZE,
                        shuffle         = True)

    print("model training is completed.")
    return history

def main():

    global train_classes
    with open('train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    global zsl_classes
    with open('zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # SET HYPERPARAMETERS

    global NUM_CLASS, NUM_ATTR, EPOCH, BATCH_SIZE
    NUM_CLASS = 27
    NUM_ATTR = 300
    BATCH_SIZE = 128
    EPOCH = 0

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # TRAINING PHASE

    (x_train, x_valid, x_zsl, x_test), (y_train, y_valid, y_zsl, y_test) = load_custom_data(dataset='awa')
    model = build_model(dataset='awa', is_fresh_model=True)
    train_model(model, (x_train, y_train), (x_valid, y_valid))
    print(model.summary())

    # ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #
    # CREATE AND SAVE ZSL MODEL

    inp         = model.input
    out         = model.layers[-2].output
    zsl_model   = Model(inp, out)
    print(zsl_model.summary())
    save_keras_model(zsl_model, model_path=MODELPATH)
    return

if __name__ == '__main__':
    main()
