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
from dataset import MyDataset, BuildDataloader, FoldTrainValDataset
from utils import compute_metrics
import torch.distributed as dist

def get_rank():
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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
rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend="nccl")
device = torch.device("cuda", local_rank)

tokenizer = BertTokenizer.from_pretrained(CFG['model']) #加载bert的分词器

class AverageMeter: #为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def train_model(model, train_loader): #训练一个epoch
    model.train() 
    
    losses = AverageMeter()
    batch_precision = AverageMeter()
    batch_recall = AverageMeter()
    batch_f1 = AverageMeter()
    
    optimizer.zero_grad()
    
    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
    
    for step, (_, input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result, gt_for_eval) in enumerate(tk):

        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
        labels_dict = [{key: value.to(device) for key, value in labels.items()} for labels in labels_dict]
        cls_reason_result = cls_reason_result.to(device)

        with autocast(): #使用半精度训练
            loss, output = model(input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result)
            scaler.scale(loss).backward()
            
            if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)): #梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                scheduler.step()

        batch = input_ids.size(0)
        precision, recall, f1 = compute_metrics(output, gt_for_eval)
        losses.update(loss.item()*CFG['accum_iter'], batch)
        batch_precision.update(precision, batch)
        batch_recall.update(recall, batch)
        batch_f1.update(f1, batch)
        
        tk.set_postfix(loss=losses.avg, f1=batch_f1.avg)
        
    return losses.avg, batch_f1.avg


def test_model(model, val_loader): #验证
    model.eval()
    
    val_losses = AverageMeter()
    val_batch_precision = AverageMeter()
    val_batch_recall = AverageMeter()
    val_batch_f1 = AverageMeter()

    
    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for step, (_, input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result, gt_for_eval) in enumerate(tk):
    
            input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
            labels_dict = [{key: value.to(device) for key, value in labels.items()} for labels in labels_dict]
            cls_reason_result = cls_reason_result.to(device)

            loss, output = model(input_ids, attention_mask, token_type_ids, labels_dict, cls_reason_result)

            batch = input_ids.size(0)
            precision, recall, f1 = compute_metrics(output, gt_for_eval)
            val_losses.update(loss.item()*CFG['accum_iter'], batch)
            val_batch_precision.update(precision, batch)
            val_batch_recall.update(recall, batch)
            val_batch_f1.update(f1, batch)
            
            tk.set_postfix(loss=val_losses.avg, f1=val_batch_f1.avg)
       
    return val_losses.avg, val_batch_f1.avg


seed_everything(CFG['seed'])

train_anno, train_labels, task_num_classes, folds = FoldTrainValDataset()

cv = [] #保存每折的最佳准确率

for fold, (trn_idx, val_idx) in enumerate(folds):
    fold += 5  #  skip the previous running
    train = [train_anno[i] for i in trn_idx]
    val = [train_anno[i] for i in val_idx]
    
    train_set = MyDataset(train, train_labels, tokenizer, vocab_type=CFG['vocab_type'], max_len=CFG['max_len'])
    val_set = MyDataset(val, train_labels, tokenizer, vocab_type=CFG['vocab_type'], max_len=CFG['max_len'])
    
    train_loader = BuildDataloader(train_set, batch_size=CFG['train_bs'], shuffle=True, num_workers=CFG['num_workers'], ddp=True)
    val_loader = BuildDataloader(val_set, batch_size=CFG['valid_bs'], shuffle=False, num_workers=CFG['num_workers'], ddp=True)
    
    best_acc = 0

    model_config = BertConfig.from_pretrained(pretrained_model_name_or_path=CFG['model'],
                                                output_hidden_states=True)
    model =  BertMultiTaskModel(config=model_config, task_num_classes=task_num_classes,
                                   model_path=CFG['model']).to(device) #模型
    sync_bn_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(sync_bn_model, device_ids=[local_rank], 
                                                    output_device=local_rank, find_unused_parameters=True)
    
    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay']) #AdamW优化器
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader)//CFG['accum_iter'], CFG['epochs']*len(train_loader)//CFG['accum_iter'])
    #get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降

    for epoch in range(CFG['epochs']):

        print('epoch:',epoch)
        time.sleep(0.2)

        train_loss, train_acc = train_model(model, train_loader)
        val_loss, val_acc = test_model(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '{}_fold_{}.pt'.format(CFG['model'].split('/')[-1], fold))
        if is_main_process():
            torch.save(model.state_dict(), '{}_fold_{}_latest.pt'.format(CFG['model'].split('/')[-1], fold))
            # print('success save only once')
    cv.append(best_acc) 
print(cv)