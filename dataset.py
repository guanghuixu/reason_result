from utils import load_json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.model_selection import StratifiedKFold
from config import CFG

task_list = CFG['task_list']
gt_list = ['cls', 'reason_type', 'reason_product', 'reason_region', 'reason_industry', \
            'result_type', 'result_product', 'result_region', 'result_industry']
empty_anno = [{'cls': np.array([0.]), 'reason_type':np.zeros([1319]), 'reason_product':np.zeros([1856]), 'reason_region':np.zeros([1510]), 'reason_industry':np.zeros([1420]), \
            'result_type':np.zeros([1299]), 'result_product':np.zeros([2324]), 'result_region':np.zeros([1378]), 'result_industry':np.zeros([1504])}]

def FoldTrainValDataset(vocab_type='most_common_1'):
    train_anno = load_json('dataset/ccks_task2_train.json')
    vocabs_anno = load_json('dataset/global_vocabs.json')[vocab_type]
    train_labels = np.load(f'dataset/train_labels.npy', allow_pickle=True).tolist()[vocab_type]
    reason_type_label = [train_labels[i['text_id']][0]['reason_type'].argmax() for i in train_anno]
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed'])\
                        .split(np.arange(len(train_anno)), reason_type_label) #五折交叉验证

    task_num_classes = {task: len(vocabs_anno[task]) for task in task_list}
    return train_anno, train_labels, task_num_classes, folds

def TestDataset(vocab_type='most_common_1'):
    test_anno = load_json('dataset/ccks_task2_eval.json')
    vocabs_anno = load_json('dataset/global_vocabs.json')[vocab_type]
    task_num_classes = {task: len(vocabs_anno[task]) for task in task_list}
    return test_anno, vocabs_anno, task_num_classes

class MyDataset(Dataset):
    def __init__(self, all_anno, labels, tokenizer, vocab_type='most_common_1', max_len=256):
        self.vocab_type = vocab_type
        self.all_anno = all_anno
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.all_anno)
    
    def __getitem__(self, idx):
        cur_sample = self.all_anno[idx]
        text_str = cur_sample["text"]
        text = self.tokenizer(text_str, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        if 'result' in cur_sample.keys():
            labels = self.labels[cur_sample["text_id"]]
        else:
            labels = []
        
        return cur_sample["text_id"], text, labels

def collate_fn(data): 
    input_ids, attention_mask, token_type_ids = [], [], []
    labels_dict = [{task: [] for task in gt_list} for _ in range(5)]
    cls_reason_result = []
    gt_for_eval = []
    text_ids = []
    for sample in data:
        text_id, text, labels = sample
        text_ids.append(text_id)
        pair_type = []
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])

        labels = labels + empty_anno * (5-len(labels))
        gt_for_eval.append(labels)
        for group,training_smple in enumerate(labels):
            for task in gt_list:
                labels_dict[group][task].append(training_smple[task])
            pair_type.append(training_smple['cls'].item())
            pair_type.append(training_smple['reason_type'].argmax())
            pair_type.append(training_smple['result_type'].argmax())
        cls_reason_result.append(pair_type)

    input_ids = torch.cat(input_ids)
    attention_mask = torch.cat(attention_mask)
    token_type_ids = torch.cat(token_type_ids)
    labels_dict = [{task: torch.tensor(labels[task]) for task in gt_list} for labels in labels_dict]
    cls_reason_result = torch.tensor(cls_reason_result).long()
    return text_ids, input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result, gt_for_eval

def BuildDataloader(dataset, batch_size, shuffle, num_workers, ddp=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)
    return dataloader
