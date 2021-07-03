import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from config import CFG
from decoder import MMT

def _get_empty_emb(cls_reason_result):
    empty_emb = torch.zeros_like(cls_reason_result)
    for i in range(5):
        empty_emb[:, i*3] = 1    # [START]
        # empty_emb[:, i*3+1] = 0  # [PAD]
        # empty_emb[:, i*3+2] = 0
    return empty_emb

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state: torch.Tensor, mask: torch.Tensor):
        q = self.fc(hidden_state).squeeze(dim=-1)
        q = q.masked_fill(mask, -np.inf)
        w = F.softmax(q, dim=-1).unsqueeze(dim=1)
        h = w @ hidden_state
        return h.squeeze(dim=1)


class AttentionClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        super(AttentionClassifier, self).__init__()
        self.attn = Attention(hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        h = self.attn(hidden_states, mask)
        out = self.fc(h)
        return out


class MultiDropout(nn.Module):

    def __init__(self, hidden_size: int, num_classes: int):
        super(MultiDropout, self).__init__()
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)])

    def forward(self, hidden_states: torch.Tensor):
        max_pool, _ = hidden_states.max(dim=1)
        avg_pool = hidden_states.mean(dim=1)
        pool = torch.cat([max_pool, avg_pool], dim=-1)
        logits = []
        for dropout in self.dropout:
            out = dropout(pool)
            out = self.fc(out)
            logits.append(out)
        logits = torch.stack(logits, dim=2).mean(dim=2)
        return logits


class BertMultiTaskModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig, task_num_classes: dict, model_path: str):
        super(BertMultiTaskModel, self).__init__(config)

        self.encoder = BertModel.from_pretrained(model_path, config=config)
        self.decoder = MMT(num_hidden_layers=4, hidden_size=config.hidden_size,num_attention_heads=config.num_attention_heads)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        classifier_fn = nn.Linear  # AttentionClassifier
        self.cls_classifier = nn.Linear(config.hidden_size, 1)
        # self.task_classifiers['common_vocabs'] = classifier_fn(config.hidden_size, task_num_classes['common_vocabs'])
        # task_num_classes.pop('common_vocabs')
        self.task_classifiers = nn.ModuleDict({task: classifier_fn(config.hidden_size, num_classes)
                                               for task, num_classes in task_num_classes.items()})
        self.reason_gru = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.result_gru = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.task_num_classes = task_num_classes
        self.loss_fn = nn.BCEWithLogitsLoss()   

    def _forward_output(self, txt_emb, txt_mask, cls_reason_result, labels=None):
        pred_empty_pair = _get_empty_emb(cls_reason_result)
        losses = []
        outputs = []
        bi_mask = torch.zeros([15, 15], dtype=torch.float).to(pred_empty_pair.device)
        reason_weights = torch.cat([self.task_classifiers['common_vocabs'].weight, 
                                    self.task_classifiers['reason_type'].weight], dim=0)
        result_weights = torch.cat([self.task_classifiers['common_vocabs'].weight, 
                                    self.task_classifiers['result_type'].weight], dim=0)
        for i in range(5):
            ii = i*3
            bi_mask[:, ii: (i+1)*3] = 1.
            _, mmt_dec_output = self.decoder(txt_emb, txt_mask, pred_empty_pair.clone(),
                    reason_weights, result_weights, bi_mask=bi_mask)
            cls, reason, result = mmt_dec_output[:, ii], mmt_dec_output[:, ii+1], mmt_dec_output[:, ii+2] 
            type_output = self._forward_type_classifier(cls, reason, result)
            output = self._forward_more_classifier(type_output, reason, result)
            outputs.append(output)
            if labels is not None:
                loss = 0.
                for iii, key in enumerate(['cls'] + CFG['task_list'][1:]):
                    loss += self.loss_fn(output[iii], labels[i][key]) / CFG['accum_iter']
                losses.append(loss)
            if self.training:  # teacher forcing
                pred_empty_pair[:, ii:(i+1)*3] = cls_reason_result[:, ii:(i+1)*3]
            else:
                # cls_pred, _, reason_type, _, result_type = type_output
                pred_empty_pair[:, ii] = (type_output[0]>=0).squeeze(-1)
                pred_empty_pair[:, ii+1] = type_output[2].argmax(dim=1)
                pred_empty_pair[:, ii+2] = type_output[4].argmax(dim=1)
        return losses, outputs

    def _forward_type_classifier(self, cls, reason, result):
        cls_pred = self.cls_classifier(cls)

        reason_hidden = self.reason_gru(reason)
        reason_type = torch.cat([
            self.task_classifiers['common_vocabs'](reason_hidden),
            self.task_classifiers["reason_type"](reason_hidden)], dim=1)

        result_hidden = self.reason_gru(result)
        result_type = torch.cat([
            self.task_classifiers['common_vocabs'](result_hidden),
            self.task_classifiers["result_type"](result_hidden)], dim=1)
        return cls_pred, reason_hidden, reason_type, result_hidden, result_type

    def _forward_more_classifier(self, type_output, reason, result):
        cls_pred, reason_hidden, reason_type, result_hidden, result_type = type_output
        reason_hidden = self.reason_gru(reason, reason_hidden)
        reason_product = torch.cat([
            self.task_classifiers['common_vocabs'](reason_hidden),
            self.task_classifiers["reason_product"](reason_hidden)], dim=1)
        reason_hidden = self.reason_gru(reason, reason_hidden)
        reason_region = torch.cat([
            self.task_classifiers['common_vocabs'](reason_hidden),
            self.task_classifiers["reason_region"](reason_hidden)], dim=1)
        reason_hidden = self.reason_gru(reason, reason_hidden)
        reason_industry = torch.cat([
            self.task_classifiers['common_vocabs'](reason_hidden),
            self.task_classifiers["reason_industry"](reason_hidden)], dim=1)

        result_hidden = self.result_gru(result, result_hidden)
        result_product = torch.cat([
            self.task_classifiers['common_vocabs'](result_hidden),
            self.task_classifiers["result_product"](result_hidden)], dim=1)
        result_hidden = self.result_gru(result, result_hidden)
        result_region = torch.cat([
            self.task_classifiers['common_vocabs'](result_hidden),
            self.task_classifiers["result_region"](result_hidden)], dim=1)
        result_hidden = self.result_gru(result, result_hidden)
        result_industry = torch.cat([
            self.task_classifiers['common_vocabs'](result_hidden),
            self.task_classifiers["result_industry"](result_hidden)], dim=1)
        return cls_pred, reason_type, reason_product, reason_region, reason_industry, \
            result_type, result_product, result_region, result_industry

    def forward(self,
                input_ids: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                labels: dict = None,
                cls_reason_result: torch.Tensor = None,
                bi_mask=None):
        # mask = input_ids == 0
        txt_emb = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        txt_emb = self.dropout(txt_emb[0])
        losses, outputs = self._forward_output(txt_emb, input_ids.ge(0), cls_reason_result, labels=labels)
        return (sum(losses), outputs)


if __name__ == '__main__':
    model_name = 'hfl/chinese-roberta-wwm-ext-large'
    bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_name,
                                                output_hidden_states=True)
    task_num_classes = {'common_vocabs':100, 'reason_type':200, 'reason_product':250, 'reason_region':300, 'reason_industry':400,
            'result_type':500, 'result_product':600, 'result_region':700, 'result_industry':800}
    model = BertMultiTaskModel(config=bert_config, task_num_classes=task_num_classes,
                                   model_path=model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    input_text = ["我叫许光辉，正在无脑测试debug，要吐了", "我叫温志全，正在无脑测试debug，要吐了"]
    max_len = 32
    input_ids, attention_mask, token_type_ids = [], [], []
    for text in input_text:
        text = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids.append(text['input_ids'])
        attention_mask.append(text['attention_mask'])
        token_type_ids.append(text['token_type_ids'])
    input_ids = torch.cat(input_ids)
    attention_mask = torch.cat(attention_mask)
    token_type_ids = torch.cat(token_type_ids)
    # labels = {'reason': torch.randn([len(input_text), 600]).softmax(-1), 'results': torch.randn([len(input_text), 700]).softmax(-1)}
    cls_reason_result = torch.tensor([[1,10,20,1,30,40,1,50,60,1,70,80,0,0,0], 
                                [1,90,100,1,110,120,1,130,140,0,0,0,0,0,0]])
    outputs = model(input_ids, token_type_ids, attention_mask, labels=None, cls_reason_result=cls_reason_result)
    print(outputs)

    