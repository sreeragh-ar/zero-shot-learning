from sklearn.preprocessing import LabelEncoder, normalize
from keras.utils import to_categorical


from pymongo import MongoClient
import json
import numpy as np

client = MongoClient()
db = client['zsl_database']
collection = db['awa']


def prepare_awa_zero_shot_data():
    zsl_classes = None
    zsl_data = list()

    with open('../zsl_classes.txt', 'r') as infile:
        zsl_classes = [str.strip(line) for line in infile]
    for class_label in zsl_classes:
        class_data = collection.find({'label_name': class_label}, {
            'feature': 1}).limit(100)
        for data in class_data:
            data_point = (data['feature'], class_label)
            zsl_data.append(data_point)
    with open('../../data/awa_zsl_data.json', 'w') as json_file:
        json.dump(zsl_data, json_file)


def prepare_awa_training_data():
    train_classes = None
    with open('../train_classes.txt', 'r') as infile:
        train_classes = [str.strip(line) for line in infile]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_classes)

    train_size = 300
    train_data = list()
    valid_data = list()
    test_data = list()

    for class_label in train_classes:
        label_one_hot = to_categorical(
            label_encoder.transform([class_label]), num_classes=27).tolist()
        class_data = collection.find({'label_name': class_label}, {
            'feature': 1}).limit(300)
        n_items = min(300,class_data.count())
        train_lim = n_items//2
        valid_lim = int((5/6)*n_items)

        count = 0
        for data in class_data:
            data_point = (data['feature'], label_one_hot)
            if count < train_lim:
                train_data.append(data_point)
                count += 1
            elif count < valid_lim:
                valid_data.append(data_point)
                count += 1
            else:
                data_point = (data['feature'], class_label)
                test_data.append(data_point)

    with open('../../data/awa_train_data.json', 'w') as json_file:
        json.dump(train_data, json_file)
    train_data = None

    with open('../../data/awa_validation_data.json', 'w') as json_file:
        json.dump(valid_data, json_file)
    valid_data = None

    with open('../../data/awa_test_data.json', 'w') as json_file:
        json.dump(test_data, json_file)


prepare_awa_training_data()
prepare_awa_zero_shot_data()
