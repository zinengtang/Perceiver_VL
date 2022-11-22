python run.py with data_root='datasets/winter21_whole/' \
gpus=4 num_nodes=1 task_cls_imagenet step200k \
per_gpu_batchsize=16 learning_rate=8e-4 num_workers=12 val_check_interval=0.2 load_pretrain=False \
model_type='PerceiverVL' latent_size_s=128 layer_drop=0.0 \
load_path=""
