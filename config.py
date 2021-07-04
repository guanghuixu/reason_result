CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2021,
    'model': 'hfl/chinese-bert-wwm-ext', #预训练模型
    'max_len': 256, #文本截断的最大长度
    'epochs': 1000,
    'train_bs': 10, #batch_size，可根据自己的显存调整
    'valid_bs': 10,
    'lr': 2e-4, # 2e-5, #学习率
    'num_workers': 8,
    'accum_iter': 2, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
    # 'vocab_type': 'most_common_1',
    "task_list": ['reason_type', 'reason_product', 'reason_region', 'reason_industry',
            'result_type', 'result_product', 'result_region', 'result_industry']

}