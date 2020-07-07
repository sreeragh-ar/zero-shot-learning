from train import load_custom_data, load_keras_model
import json
import numpy as np
from sklearn.neighbors import KDTree

train_classes = None
zsl_classes = None
with open('train_classes.txt', 'r') as infile:
    train_classes = [str.strip(line) for line in infile]
with open('zsl_classes.txt', 'r') as infile:
    zsl_classes = [str.strip(line) for line in infile]
classes_considered = train_classes + zsl_classes
class_vectors = None
with open('../data/vectors_data.json') as json_file:
    class_vectors = json.load(json_file)
training_vectors = sorted([(vector_data['label'], vector_data['vector'])
                           for vector_data in class_vectors if vector_data['label'] in classes_considered], key=lambda x: x[0])
classnames, vectors = zip(*training_vectors)
vectors = np.asarray(vectors, dtype=np.float)

zsl_model = load_keras_model(model_path='../model/backup/')

(x_train, x_valid, x_zsl, x_test), (y_train, y_valid,
                            y_zsl, y_test) = load_custom_data(dataset='awa')

tree = KDTree(vectors)
pred_zsl = zsl_model.predict(x_zsl)

top5, top3, top1 = 0, 0, 0
for i, pred in enumerate(pred_zsl):
    pred = np.expand_dims(pred, axis=0)
    dist_5, index_5 = tree.query(pred, k=5)
    pred_labels = [classnames[index] for index in index_5[0]]
    true_label = y_zsl[i]
    if true_label in pred_labels:
        top5 += 1
    if true_label in pred_labels[:3]:
        top3 += 1
    if true_label in pred_labels[0]:
        top1 += 1

print()
print("ZERO SHOT CLASS SCORES")
print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_zsl))))
print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_zsl))))
print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_zsl))))

pred_test = zsl_model.predict(x_test)

top5, top3, top1 = 0, 0, 0
for i, pred in enumerate(pred_test):
    pred = np.expand_dims(pred, axis=0)
    dist_5, index_5 = tree.query(pred, k=5)
    pred_labels = [classnames[index] for index in index_5[0]]
    true_label = y_test[i]
    if true_label in pred_labels:
        top5 += 1
    if true_label in pred_labels[:3]:
        top3 += 1
    if true_label in pred_labels[0]:
        top1 += 1

print()
print("TRAINING CLASS SCORES")
print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_test))))
print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_test))))
print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_test))))
