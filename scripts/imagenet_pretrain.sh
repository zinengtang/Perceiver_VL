python run.py with data_root='datasets/winter21_whole/' \
gpus=[0,1] num_nodes=1 task_cls_imagenet step200k \
per_gpu_batchsize=50 learning_rate=8e-4 num_workers=16 val_check_interval=0.33 load_pretrain=False \
model_type='PerceiverVL_version0' latent_size_s=128 layer_drop=0.0 random_proj=1 \
# load_path="result/imagenet_pretrain_v2/version_0/checkpoints/last.ckpt"