from pymongo import MongoClient
import json

client = MongoClient()
db = client['zsl_database']
collection = db['class_vectors']

with open('../../data/vectors_data.json') as json_file:
    vectors_data = json.load(json_file)
    print(vectors_data[0])

collection.insert_many(vectors_data)