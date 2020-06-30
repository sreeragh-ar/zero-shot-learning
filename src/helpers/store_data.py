from pymongo import MongoClient
import random
import os

LABEL_FILE = 'awa_classes.txt'
client = MongoClient()
db = client['zsl_database']
collection = db['awa']

labels = [None]
line_no = 1
with open(LABEL_FILE) as label_cursor:
    for label_data in label_cursor:
        label_num, label = label_data.strip().split()
        labels.append(label)

# convert feature to list, add label_names
data = collection.find()
for data_point in data:
    _id = data_point['_id']
    update = {}
    if not isinstance(data_point['feature'], list):
        feature_vector = list(map(float, data_point['feature'].split()))
        update['feature'] = feature_vector
    if 'label_name' not in data_point:
        update['label_name'] = labels[data_point['label']]
    if update:
        collection.update({'_id': _id}, {'$set': update})
