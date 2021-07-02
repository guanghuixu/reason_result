# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
from torch import nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer

from pythia.modules.encoders import ImageEncoder

import random


@registry.register_model("m4c")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        self.remove_ocr_fasttext = getattr(
            self.config.ocr, 'remove_ocr_fasttext', False
        )
        self.remove_ocr_phoc = getattr(
            self.config.ocr, 'remove_ocr_phoc', False
        )
        self.remove_ocr_frcn = getattr(
            self.config.ocr, 'remove_ocr_frcn', False
        )
        self.remove_ocr_semantics = getattr(
            self.config.ocr, 'remove_ocr_semantics', False
        )
        self.remove_ocr_bbox = getattr(
            self.config.ocr, 'remove_ocr_bbox', False
        )

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        if self.training and self.config.using_unpair_sample:
            results['unpair_txt_scores'] = fwd_results["unpair_txt_scores"]
            results['unpair_img_scores'] = fwd_results["unpair_img_scores"]
        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        fwd_results['txt_inds'] = sample_list.text

        # binary mask of valid text (question words) vs padding
        fwd_results['txt_mask'] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )

        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out)

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(obj_feat)
            ) + self.obj_bbox_layer_norm(
                self.linear_obj_bbox_to_mmt_in(obj_bbox)
            )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, :ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        if self.remove_ocr_fasttext:
            ocr_fasttext = torch.zeros_like(ocr_fasttext)
        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors],
            dim=-1
        )
        ocr_bbox = sample_list.ocr_bbox_coordinates
        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
                self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
            )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results['ocr_mask'] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_mmt(self, sample_list, fwd_results, txt='', obj='', ocr=''):
        mmt_results = self.mmt(
            txt_emb=fwd_results[txt+'txt_emb'],
            txt_mask=fwd_results[txt+'txt_mask'],
            obj_emb=fwd_results[obj+'obj_mmt_in'],
            obj_mask=fwd_results[obj+'obj_mask'],
            ocr_emb=fwd_results[ocr+'ocr_mmt_in'],
            ocr_mask=fwd_results[ocr+'ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results, key='scores'):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results[key] = scores

    def _generate_unpair_samples(self, sample_list, fwd_results):
        txt = obj = ocr = 'unpair'
        batch_size = fwd_results['txt_emb'].size(0)
        index_txt = random.sample(range(0, batch_size), batch_size)
        index_img = random.sample(range(0, batch_size), batch_size)

        fwd_results[txt+'txt_emb'] = fwd_results['txt_emb'][index_txt]
        fwd_results[txt+'txt_mask'] = fwd_results['txt_mask'][index_txt]
        fwd_results[txt+'index_txt'] = index_txt

        fwd_results[obj+'obj_mmt_in'] = fwd_results['obj_mmt_in'][index_img]
        fwd_results[obj+'obj_mask'] = fwd_results['obj_mask'][index_img]
        fwd_results[ocr+'ocr_mmt_in'] = fwd_results['ocr_mmt_in'][index_img]
        fwd_results[ocr+'ocr_mask'] = fwd_results['ocr_mask'][index_img]
        fwd_results[ocr+'index_img'] = index_img

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
            if self.config.using_unpair_sample:
                self._generate_unpair_samples(sample_list, fwd_results)
                # forward_unpair_txt
                self._forward_mmt(sample_list, fwd_results, txt='unpair', obj='', ocr='')
                self._forward_output(sample_list, fwd_results, key='unpair_txt_scores')
                # forward_unpair_image
                self._forward_mmt(sample_list, fwd_results, txt='', obj='unpair', ocr='unpair')
                self._forward_output(sample_list, fwd_results, key='unpair_img_scores')
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, num_hidden_layers=4,hidden_size=768,num_attention_heads=12):
        config = BertConfig(num_hidden_layers=num_hidden_layers, 
                            hidden_size=hidden_size,
                            num_attention_heads=num_attention_heads)
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.generate_mask = _get_generate_mask(seq_len=3*5)
        self.init_weights()

    def forward(self,
                txt_emb,
                txt_mask,
                cls_reason_result,
                reason_weight,
                result_weight,
                bi_mask=None):

        pair_embs = self.prev_pred_embeddings(cls_reason_result, reason_weight, result_weight)

        pair_mask = torch.zeros(
            pair_embs.size(0),
            pair_embs.size(1),
            dtype=torch.float32,
            device=pair_embs.device
        )
        

        encoder_inputs = torch.cat(
            [txt_emb, pair_embs],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, pair_mask],
            dim=1
        )

        txt_max_num = txt_mask.size(-1)
        dec_max_num = pair_mask.size(-1)

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        if bi_mask is not None:
            reasoning_mask = bi_mask
        else:
            reasoning_mask = self.generate_mask.to(pair_embs.device)
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = reasoning_mask

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, :txt_max_num]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]
        
        return mmt_txt_output, mmt_dec_output


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_TYPE_NUM = 3  # cls, reason, result
        MAX_DEC_GROUP = 5
        MAX_DEC_LENGTH = MAX_TYPE_NUM * MAX_DEC_GROUP
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps
        
        # 1: exist; 0: None
        self.cls_embeddings = nn.Embedding(2, config.hidden_size)

        self.token_type_embeddings = nn.Parameter(torch.randn(1, MAX_TYPE_NUM, hidden_size))
        torch.nn.init.xavier_uniform_(self.token_type_embeddings)

        self.position_embeddings = nn.Parameter(torch.randn(1, MAX_DEC_LENGTH, hidden_size))
        torch.nn.init.xavier_uniform_(self.position_embeddings)

        self.pair_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, cls_reason_results, reason_weight, result_weight):
        seq_length  = cls_reason_results.size(1)
        raw_pair_embs = []
        embeddings_map =  [self.cls_embeddings, reason_weight, result_weight]
        for column in range(seq_length):
            batch_item = cls_reason_results[:, column]
            fn_ids = column%3
            if fn_ids == 0:
                batch_process = embeddings_map[fn_ids](batch_item)
            else:
                batch_process = embeddings_map[fn_ids][batch_item]
            raw_pair_embs.append(batch_process)
        raw_pair_embs = torch.stack(raw_pair_embs, dim=1)
        raw_pair_embs = self.pair_layer_norm(raw_pair_embs)

        position_embeddings = self.position_embeddings.expand_as(raw_pair_embs)
        token_type_embeddings = self.token_type_embeddings.repeat(1, 5, 1).expand_as(raw_pair_embs)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)

        pair_embs = raw_pair_embs + embeddings

        return pair_embs


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask

def _get_generate_mask(seq_len):
    new_mask = []
    for i in range(1, seq_len+1, 3):
        new_mask.append([1.0] * i + [0.0] * (seq_len - i))
        new_mask.append([1.0] * (i+2) + [0.0] * (seq_len - (i+2)))
        new_mask.append([1.0] * (i+2) + [0.0] * (seq_len - (i+2)))
    return torch.tensor(new_mask)


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

if __name__ == '__main__':
    txt_emb = torch.randn([2, 5, 768])
    txt_mask = torch.tensor([[1,1,1,0,0], [1,1,0,0,0]])
    cls_reason_result = torch.tensor([[1,10,20,1,30,40,1,50,60,1,70,80,0,0,0], 
    [1,90,100,1,110,120,1,130,140,0,0,0,0,0,0]])
    config = BertConfig(num_hidden_layers=4, hidden_size=768)
    model = MMT(config)
    reason_classifier = nn.Linear(768, 200)
    result_classifier = nn.Linear(768, 300)
    output = model(txt_emb,
                txt_mask,
                cls_reason_result,
                reason_classifier.weight,
                result_classifier.weight,
                bi_mask=None)
    print(output.keys())
    


