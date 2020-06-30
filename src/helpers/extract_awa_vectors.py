'''
Original file is located at
    https://colab.research.google.com/drive/1hKj_SijC2JJzdmLIx0tMzYqrFjhhQpdO
'''
import json
from google.colab import drive
drive.mount('/content/drive')

import gensim.models.keyedvectors as word2vec
model = word2vec.KeyedVectors.load_word2vec_format('/content/drive/My Drive/Colab Notebooks/GoogleNews-vectors-negative300.bin', binary=True)
word2vec_data = []
with open('/content/drive/My Drive/Colab Notebooks/awa_classes.txt') as classes_cursor:
    for line in classes_cursor:
        label_num, label = line.strip().split()
        label_obj = {'label': label}
        label_key = label.replace('+', '_').capitalize()
        label_obj['vector_key'] = label_key
        try:
          label_obj['vector'] = model[label_key].tolist()
          word2vec_data.append(label_obj)
        except:
          print('Breaking', label, label_key)

label_obj = {'label': 'persian+cat'}
label_obj['vector_key'] = 'longhaired_cat'
label_obj['vector'] =  model[label_obj['vector_key']].tolist()
word2vec_data.append(label_obj)

label_obj = {'label': 'blue+whale'}
label_obj['vector_key'] = 'baleen_whale'
label_obj['vector'] =  model[label_obj['vector_key']].tolist()
word2vec_data.append(label_obj)

label_obj = {'label': 'spider+monkey'}
label_obj['vector_key'] = 'Spider_Monkey'
label_obj['vector'] =  model[label_obj['vector_key']].tolist()
word2vec_data.append(label_obj)

with open('/content/drive/My Drive/Colab Notebooks/vectors_data.json', 'w') as json_file:
  json.dump(word2vec_data, json_file)
