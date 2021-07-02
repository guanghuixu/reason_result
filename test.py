import json 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
# from sklearn.model_selection import *
from transformers import *
from config import CFG
from model import BertMultiTaskModel
from dataset import MyDataset, BuildDataloader, TestDataset
from utils import compute_metrics, load_json


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained(CFG['model']) #加载bert的分词器

seed_everything(CFG['seed'])

test_anno, task_num_classes = TestDataset()
test_set = MyDataset(test_anno, labels=None, tokenizer=tokenizer, vocab_type=CFG['vocab_type'], max_len=CFG['max_len'])
test_loader = BuildDataloader(test_set, batch_size=CFG['val_bs'], shuffle=False, num_workers=CFG['num_workers'], ddp=False)

model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=CFG['model'],
                                                output_hidden_states=True)
model =  BertMultiTaskModel(config=model_config, task_num_classes=task_num_classes,
                                   model_path=CFG['model']).to(device) #模型

predictions = []

for fold in [0,1,2,3,4]: #把训练后的五个模型挨个进行预测
    fold = 0  # debug
    y_pred = []
    model.load_state_dict(torch.load('{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold)))
    
    with torch.no_grad():
        tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True)
        for step, (input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result, gt_for_eval) in enumerate(tk):
    
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            # labels_dict = [{key: value.to(device) for key, value in labels.items()} for labels in labels_dict]
            cls_reason_result = cls_reason_result.to(device)

            _, output = model(input_ids, attention_mask, token_type_ids, labels=None, cls_reason_result=cls_reason_result)
            
    predictions += [y_pred]

predictions = np.mean(predictions,0).argmax(1) #将结果按五折进行平均，然后argmax得到label



sub = pd.read_csv('sample.csv',dtype=object) #提交
sub['label'] = predictions
sub['label'] = sub['label'].apply(lambda x:['A','B','C','D'][x])

sub.to_csv('sub01.csv',index=False)