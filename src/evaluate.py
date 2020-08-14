from train import load_custom_data, load_keras_model
import json
import numpy as np
import argparse
import os
from sklearn.neighbors import KDTree

def get_harmonic_mean(a, b):
    return (2 * (a * b)/(a + b))

def get_arguments():
    parser = argparse.ArgumentParser(description='Data details')
    parser.add_argument("--dataset", default='awa',
                        help="The dataset to be used for training")
    return parser.parse_args()

args = get_arguments()
DATASET_SPLITS_DIR = os.path.join('dataset_splits', args.dataset)
DATA_DIR = os.path.join('..', 'data')

train_classes = None
zsl_classes = None

with open(os.path.join(DATASET_SPLITS_DIR,'train_classes.txt'), 'r') as infile:
    train_classes = [str.strip(line) for line in infile]
with open(os.path.join(DATASET_SPLITS_DIR,'zsl_classes.txt'), 'r') as infile:
    zsl_classes = [str.strip(line) for line in infile]
classes_considered = train_classes + zsl_classes
class_vectors = None
file_name = 'vectors_data.json'
if args.dataset != 'awa':
    file_name = f'{args.dataset}_{file_name}'
vectors_file_path = os.path.join(DATA_DIR, file_name)
with open(vectors_file_path) as json_file:
    class_vectors = json.load(json_file)
training_vectors = sorted([(vector_data['label'], vector_data['vector'])
                           for vector_data in class_vectors if vector_data['label'] in classes_considered], key=lambda x: x[0])
classnames, vectors = zip(*training_vectors)
vectors = np.asarray(vectors, dtype=np.float)

zsl_model = load_keras_model(model_path='../model/backup_models/lad_6000/')

(x_train, x_valid, x_zsl, x_test), (y_train, y_valid,
                            y_zsl, y_test) = load_custom_data(args.dataset)

tree = KDTree(vectors)
pred_zsl = zsl_model.predict(x_zsl)

top5, top3, top1 = 0, 0, 0
scores = {}
for class_name in classes_considered:
    scores[class_name] = {'top1': 0, 'top3':0, 'top5': 0, 'total': 0}
for i, pred in enumerate(pred_zsl):
    pred = np.expand_dims(pred, axis=0)
    dist_5, index_5 = tree.query(pred, k=5)
    pred_labels = [classnames[index] for index in index_5[0]]
    true_label = y_zsl[i]
    scores[true_label]['total'] += 1
    if true_label in pred_labels:
        top5 += 1
        scores[true_label]['top5'] += 1
    if true_label in pred_labels[:3]:
        top3 += 1
        scores[true_label]['top3'] += 1
    if true_label in pred_labels[0]:
        top1 += 1
        scores[true_label]['top1'] += 1

print()
print("ZERO SHOT CLASS SCORES")
print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_zsl))))
print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_zsl))))
print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_zsl))))
top1_total_accuracy = 0
top3_total_accuracy = 0
top5_total_accuracy = 0
for class_name in zsl_classes:
    top1_accuracy = scores[class_name]['top1'] / float(scores[class_name]['total'])
    top1_total_accuracy += top1_accuracy
    top3_accuracy = scores[class_name]['top3'] / float(scores[class_name]['total'])
    top3_total_accuracy += top3_accuracy
    top5_accuracy = scores[class_name]['top5'] / float(scores[class_name]['total'])
    top5_total_accuracy += top5_accuracy
avg_per_zsl_class_top1_acc = top1_total_accuracy / float(len(zsl_classes))
avg_per_zsl_class_top3_acc = top3_total_accuracy / float(len(zsl_classes))
avg_per_zsl_class_top5_acc = top5_total_accuracy / float(len(zsl_classes))

print("-> Per-class-Top-1 %.2f" % (avg_per_zsl_class_top1_acc))
print("-> Per-class-Top-3 %.2f" % (avg_per_zsl_class_top3_acc))
print("-> Per-class-Top-5 %.2f" % (avg_per_zsl_class_top5_acc))
pred_test = zsl_model.predict(x_test)

top5, top3, top1 = 0, 0, 0
for i, pred in enumerate(pred_test):
    pred = np.expand_dims(pred, axis=0)
    dist_5, index_5 = tree.query(pred, k=5)
    pred_labels = [classnames[index] for index in index_5[0]]
    true_label = y_test[i]
    scores[true_label]['total'] += 1
    if true_label in pred_labels:
        top5 += 1
        scores[true_label]['top5'] += 1
    if true_label in pred_labels[:3]:
        top3 += 1
        scores[true_label]['top3'] += 1
    if true_label in pred_labels[0]:
        top1 += 1
        scores[true_label]['top1'] += 1

print()
print("TRAINING CLASS SCORES")
print("-> Top-5 Accuracy: %.2f" % (top5 / float(len(x_test))))
print("-> Top-3 Accuracy: %.2f" % (top3 / float(len(x_test))))
print("-> Top-1 Accuracy: %.2f" % (top1 / float(len(x_test))))

top1_total_accuracy = 0
top3_total_accuracy = 0
top5_total_accuracy = 0
for class_name in train_classes:
    top1_accuracy = scores[class_name]['top1'] / float(scores[class_name]['total'])
    top1_total_accuracy += top1_accuracy
    top3_accuracy = scores[class_name]['top3'] / float(scores[class_name]['total'])
    top3_total_accuracy += top3_accuracy
    top5_accuracy = scores[class_name]['top5'] / float(scores[class_name]['total'])
    top5_total_accuracy += top5_accuracy
avg_per_train_class_top1_acc = top1_total_accuracy / float(len(train_classes))
avg_per_train_class_top3_acc = top3_total_accuracy / float(len(train_classes))
avg_per_train_class_top5_acc = top5_total_accuracy / float(len(train_classes))


print("-> Per-class-Top-1 %.2f" % (avg_per_train_class_top1_acc))
print("-> Per-class-Top-3 %.2f" % (avg_per_train_class_top3_acc))
print("-> Per-class-Top-5 %.2f" % (avg_per_train_class_top5_acc))


generalized_per_class_top1_acc = get_harmonic_mean(
    avg_per_train_class_top1_acc, avg_per_zsl_class_top1_acc)
generalized_per_class_top3_acc = get_harmonic_mean(
    avg_per_train_class_top3_acc, avg_per_zsl_class_top3_acc)
generalized_per_class_top5_acc = get_harmonic_mean(
    avg_per_train_class_top5_acc, avg_per_zsl_class_top5_acc)

print("-> Generalized-per-class-Top-1 %.2f" % (generalized_per_class_top1_acc))
print("-> Generalized-per-class-Top-3 %.2f" % (generalized_per_class_top3_acc))
print("-> Generalized-per-class-Top-5 %.2f" % (generalized_per_class_top5_acc))