#coding=utf-8
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

import os
import json
from collections import Counter

from torch.nn.modules import activation
from config import CFG
import torch
import numpy as np

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
    
def send_emails(subject="Recording experiments logs", content="Anything you want to write", log_file=None,
            msg_from='1643206826@qq.com', msg_to='1643206826@qq.com'):
    with open('/mnt/cephfs/home/xuguanghui/password.txt') as f:
        passwd=f.readlines()[0].strip()                          

    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = msg_from
    msg['To'] = msg_to
    s = smtplib.SMTP_SSL("smtp.qq.com", 465)
    s.login(msg_from, passwd)

    if log_file:
        log_txt = MIMEApplication(open(log_file,'rb').read())
        log_txt.add_header('Content-Disposition', 'attachment', filename='exp_log.log')
        multipart = MIMEMultipart()
        multipart.attach(msg)
        multipart.attach(log_txt)
        s.sendmail(msg_from, msg_to, multipart.as_bytes())
    else:
        s.sendmail(msg_from, msg_to, msg.as_string())


class Generation:
    def __init__(self, vocabs_anno):
        self.vocabs_anno = vocabs_anno
        self.sigmoid = torch.nn.Sigmoid()

    def obtain_multi_idx(self, value, theta=0.5):
        return np.where(value.cpu().numpy()>=theta)[0]
    
    def generate_token(self, multi_idx, flag):
        tokens = []
        # print(multi_idx)
        for idx in multi_idx:
            tokens.append(self.vocabs_anno[flag][idx])
        return ','.join(tokens)
    
    def generate_batch(self, text_ids, output):
        batch_size = output[0][0].size(0)
        batch_generations = []
        for batch in range(batch_size):
            text_id_dict = {"text_id": text_ids[batch], "result": []}
            for group in range(5):
                group_dict = {}
                cls_pred, reason_type, reason_product, reason_region, reason_industry, \
                        result_type, result_product, result_region, result_industry = output[group]
                if self.sigmoid(cls_pred[batch]) < 0.5 and len(text_id_dict["result"])>0:
                    continue
                # print(text_ids[batch])
                group_dict["reason_type"] = self.generate_token([reason_type[batch].argmax().item()], flag="type")
                group_dict["result_type"] = self.generate_token([result_type[batch].argmax().item()], flag="type")
                # multi_label_prediction
                group_dict["reason_product"] = self.generate_token(
                    self.obtain_multi_idx(reason_product[batch]), flag="product")
                group_dict["reason_region"] = self.generate_token(
                    self.obtain_multi_idx(reason_region[batch]), flag="region")
                group_dict["reason_industry"] = self.generate_token(
                    self.obtain_multi_idx(reason_industry[batch]), flag="industry")
                group_dict["result_product"] = self.generate_token(
                    self.obtain_multi_idx(result_product[batch]), flag="product")
                group_dict["result_region"] = self.generate_token(
                    self.obtain_multi_idx(result_region[batch]), flag="region")
                group_dict["result_industry"] = self.generate_token(
                    self.obtain_multi_idx(result_industry[batch]), flag="industry")
                text_id_dict["result"].append(group_dict)
            batch_generations.append(text_id_dict)
        return batch_generations

def transform_output2batch(output):
    sigmoid = torch.nn.Sigmoid()
    def ge_threshold(value, theta=0.5):
        return (sigmoid(value)>=theta).cpu().numpy()

    batch_size = output[0][0].size(0)
    batch_predictions = [{} for _ in range(batch_size)]
    for batch in range(batch_size):
        for group in range(5):
            cls_pred, reason_type, reason_product, reason_region, reason_industry, \
                    result_type, result_product, result_region, result_industry = output[group]
            if sigmoid(cls_pred[batch]) < 0.5:
                continue
            key = f"{reason_type[batch].argmax()}_{result_type[batch].argmax()}"
            value = [ge_threshold(reason_product[batch]),
                    ge_threshold(reason_region[batch]),
                    ge_threshold(reason_industry[batch]),
                    ge_threshold(result_product[batch]), 
                    ge_threshold(result_region[batch]),
                    ge_threshold(result_industry[batch])]
            batch_predictions[batch][key] = value
    return batch_predictions

def compute_metrics(output, batch_label: dict):
    # top_keys = ['reason_type', 'result_type']
    fine_grained = ['reason_product', 'reason_region', 'reason_industry',
            'result_product', 'result_region', 'result_industry']
    batch_predictions = transform_output2batch(output)
    batch_precision = batch_recall = batch_f1 = 0
    batch_size = len(batch_label)
    for batch, results in enumerate(batch_label):
        pred = batch_predictions[batch]
        # pred_keys = list(pred.keys())
        anno_gt_num = anno_pred_num = correct_num = 0
        for result in results:
            key = f"{result['reason_type'].argmax()}_{result['result_type'].argmax()}"
            # pred[key] = pred[pred_keys[0]]
            if result['cls'] == 1 and key in pred.keys(): # 预测的三元组是对的
                cur_pred = pred.pop(key)
                anno_pred_num = 0
                for item in cur_pred:
                    # item[0] = 0 # remove padding
                    item[1] = 0 # remove ''
                    anno_pred_num += item.sum()
                for i, each_item in enumerate(fine_grained):
                    result[each_item][1] = 0
                    anno_gt_num += result[each_item].sum()
                    correct_num += (result[each_item]*cur_pred[i]).sum()
            elif result['cls'] == 1: # 预测不对，只统计gt的标注
                for i, each_item in enumerate(fine_grained):
                    result[each_item][1] = 0
                    anno_gt_num += result[each_item].sum()
        anno_pred_num = 0
        for values in pred.values():
            for each_pred_item in values:
                each_pred_item[1] = 0
                anno_pred_num += each_pred_item.sum()
        precision = correct_num / max(anno_pred_num, 1)
        recall = correct_num / max(anno_gt_num, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1)
        batch_precision += precision
        batch_recall += recall
        batch_f1 +=f1
    return batch_precision/batch_size, batch_recall/batch_size, batch_f1/batch_size

if __name__ == '__main__':
    # send_emails()
    output, batch_label = torch.load('./debug.pt')
    p, r, f1 = compute_metrics(output, batch_label)
    print(p, r, f1)