# CUDA_VISIBLE_DEVICES=$1 python train.py
CUDA_VISIBLE_DEVICES=$1 python -m torch.distributed.launch --nproc_per_node=$2 --master_port=22222 ddp_train.py
