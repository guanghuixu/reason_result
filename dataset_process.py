import os
import json
from collections import Counter
import numpy as np
import pandas as pd

def txt2json(file):
    data_list = []
    with open(file) as f:
        data = f.readlines()
        for i in data:
            data_list.append(i.replace('\n', ','))
    with open(file.replace('.txt','.json').replace('/ori',''), 'w+') as f:
        f.write('[')
        f.writelines(data_list)
        f.write(']')
    
def load_json(file):
    with open(file) as f:
        return json.load(f)
        

def save_json(value, file):
    with open(file, 'w+', encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)


# In[2]:


# # only need perform once
# txt2json('../dataset/ori/ccks_task2_train.txt')
train_data = load_json('dataset/ccks_task2_train.json')
# txt2json('../dataset/ori/ccks_task2_eval_data.txt')
eval_data = load_json('dataset/ccks_task2_eval.json')

names_list = ['reason_type', 'reason_product', 'reason_region', 'reason_industry',
            'result_type', 'result_product', 'result_region', 'result_industry']
anno_dict = {key: [] for key in names_list}
all_vocabs = []

for i in train_data:
    results = i['result']
    for result in results:
        for key,value in result.items():
            anno_dict[key] = anno_dict[key] + value.split(',')
        
for key,value in anno_dict.items():
    new_value = list(set(value))
    anno_dict[key] = ['<PAD>'] + new_value
    print(key, len(anno_dict[key]))
    all_vocabs = all_vocabs + new_value

save_json(anno_dict, 'dataset/split_task_vocab.json')

new_all_vocabs = ['<PAD>'] + list(set(all_vocabs))
vocabs_json = {'all_vocabs': new_all_vocabs}
print('all_vocabs: {}'.format(len(vocabs_json['all_vocabs'])))
counter_all_vocabs = Counter(all_vocabs)
for i in range(1,4):  # i=0 i.e. all_vocabs
    most_common = 'most_common_{}'.format(i)
    common_vocabs = {key for key,value in counter_all_vocabs.items() if value>i}
    vocabs_json[most_common] = {'common_vocabs': ['<PAD>'] + list(common_vocabs)}
    print(most_common + ' :{}'.format(len(common_vocabs)))
    for name in names_list:
        vocabs_json[most_common][name] = list(set(anno_dict[name])-common_vocabs) # list(common_vocabs&anno_dict[name]^anno_dict[name])
        print(most_common + '_{} :{}'.format(name, len(vocabs_json[most_common][name])))
#         assert len(anno_dict[name]) == len(vocabs_json[most_common][name]) + len(anno_dict[name]&common_vocabs)

save_json(vocabs_json, 'dataset/global_vocabs.json')

train_anno = load_json('dataset/ccks_task2_train.json')
vocabs_anno = load_json('dataset/global_vocabs.json')
labels_anno = {}
vocab_types = ['all_vocabs', 'most_common_1', 'most_common_2', 'most_common_3']

for vocab_type in vocab_types[1:]:
    vocab = vocabs_anno[vocab_type]
    labels_anno[vocab_type] = {}
    common_num = len(vocab['common_vocabs'])
    for anno in train_anno:
        labels_anno[vocab_type][anno['text_id']] = []
        results = anno['result']
        for i,result in enumerate(results):
            tmp_dict = {}
            for task, labels in result.items():
                tmp_dict[task] = np.zeros(common_num + len(vocab[task]))
                labels = labels.split(',')
                for label in labels:
                    if label in vocab['common_vocabs']:
                        idx = vocab['common_vocabs'].index(label)
                    else:
                        idx = common_num + vocab[task].index(label)
                    tmp_dict[task][idx] = 1
            tmp_dict['cls'] = np.array([1.0])
            labels_anno[vocab_type][anno['text_id']].append(tmp_dict)
            # reason_result_pair = "{}_{}".format(tmp_dict['reason_type'].argmax(), tmp_dict['result_type'].argmax())
            # labels_anno[vocab_type][anno['text_id']].append({reason_result_pair: tmp_dict})

vocab_type = 'all_vocabs'
all_vocabs = vocabs_anno[vocab_type]
labels_anno['all_vocabs'] = {}
common_num = len(all_vocabs)
for anno in train_anno:
    labels_anno[vocab_type][anno['text_id']] = []
    results = anno['result']
    for result in results:
        tmp_dict = {}
        for task, labels in result.items():
            tmp_dict[task] = np.zeros(common_num)
            labels = labels.split(',')
            for label in labels:
                idx = all_vocabs.index(label)
                tmp_dict[task][idx] = 1
        tmp_dict['cls'] = np.array([1.0])
        labels_anno[vocab_type][anno['text_id']].append(tmp_dict)
        # reason_result_pair = "{}_{}".format(tmp_dict['reason_type'].argmax(), tmp_dict['result_type'].argmax())
        # labels_anno[vocab_type][anno['text_id']].append({reason_result_pair: tmp_dict})


with open('./dataset/train_labels.npy', 'wb') as f:
    np.save(f, labels_anno)
# np.load('./dataset/train_labels.npy', allow_pickle=True).tolist()