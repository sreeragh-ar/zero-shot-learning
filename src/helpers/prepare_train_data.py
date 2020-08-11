from sklearn.preprocessing import LabelEncoder, normalize
from keras.utils import to_categorical


from pymongo import MongoClient
import json
import numpy as np
import os
import argparse

client = MongoClient()
db = client['zsl_database']
collection = db['awa']

DATA_SPLITS_DIR = os.path.join('..', 'splits')
DATA_DUMP_DIR = os.path.join('..', '..', 'data')


def prepare_zero_shot_data(dataset='awa'):
    zsl_classes = None
    zsl_data = list()
    file_name = 'zsl_classes.txt'
    classes_file_path = os.path.join(DATA_SPLITS_DIR, dataset, file_name)
    with open(classes_file_path, 'r') as zsl_classes:
        for line in zsl_classes:
            class_label = str.strip(line)
            class_data = collection.find({'label_name': class_label}, {
                'feature': 1}).limit(100)
            print('Collecting data of', class_label)
            for data in class_data:
                data_point = (data['feature'], class_label)
                zsl_data.append(data_point)
    dump_file_path = os.path.join(DATA_DUMP_DIR, f'{dataset}_zsl_data.json')
    with open(dump_file_path, 'w') as json_file:
        json.dump(zsl_data, json_file)


def prepare_training_data(dataset='awa'):
    train_classes = None
    file_name = 'train_classes.txt'
    classes_file_path = os.path.join(DATA_SPLITS_DIR, dataset, file_name)
    with open(classes_file_path, 'r') as infile:
        train_classes = [str.strip(line) for line in infile]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_classes)

    train_size = 300
    train_data = list()
    valid_data = list()
    test_data = list()
    num_classes = len(train_classes)
    for class_label in train_classes:
        print('Collecting data of', class_label)
        label_one_hot = to_categorical(
            label_encoder.transform([class_label]), num_classes).tolist()
        class_data = collection.find({'label_name': class_label}, {
            'feature': 1}).limit(train_size)
        n_items = min(train_size, class_data.count())
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
    dump_file_path = os.path.join(
        DATA_DUMP_DIR, f'{dataset}_train_data.json')
    with open(dump_file_path, 'w') as json_file:
        json.dump(train_data, json_file)
    train_data = None

    dump_file_path = os.path.join(
        DATA_DUMP_DIR, f'{dataset}_validation_data.json')
    with open(dump_file_path, 'w') as json_file:
        json.dump(valid_data, json_file)
    valid_data = None

    dump_file_path = os.path.join(
        DATA_DUMP_DIR, f'{dataset}_test_data.json')
    with open(dump_file_path, 'w') as json_file:
        json.dump(test_data, json_file)


def get_arguments():
    parser = argparse.ArgumentParser(description='Data details')
    parser.add_argument("--dataset", default='awa',
                        help="The dataset to be used for training")
    return parser.parse_args()


def main():
    global args, collection
    args = get_arguments()
    if args.dataset != 'awa':
        collection = db[args.dataset]
    prepare_zero_shot_data(args.dataset)
    prepare_training_data(args.dataset)


if __name__ == '__main__':
    main()
