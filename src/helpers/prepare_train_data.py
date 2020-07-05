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
            'feature': 1}).limit(400)
        count = 0
        print('Getting', class_label)
        for data in class_data:
            data_point = (data['feature'], class_label)
            if count >= 300:
                break
            zsl_data.append(data_point)
            count += 1
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

    for class_label in train_classes:
        label_one_hot = to_categorical(
            label_encoder.transform([class_label]), num_classes=27).tolist()
        class_data = collection.find({'label_name': class_label}, {
            'feature': 1}).limit(400)
        count = 0
        for data in class_data:
            data_point = (data['feature'], label_one_hot)
            if count < 300:
                train_data.append(data_point)
                count += 1
            else:
                valid_data.append(data_point)

    with open('../../data/awa_train_data.json', 'w') as json_file:
        json.dump(train_data, json_file)
    train_data = None

    with open('../../data/awa_validation_data.json', 'w') as json_file:
        json.dump(valid_data, json_file)

prepare_awa_training_data()
prepare_awa_zero_shot_data()
