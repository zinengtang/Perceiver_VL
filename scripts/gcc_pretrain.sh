python run.py with data_root='datasets/cc/' gpus=3 num_nodes=1 task_mlm_itm step200k \
per_gpu_batchsize=24 model_type='PerceiverVL_version0' learning_rate=3e-4 \
draw_false_image=1 latent_size_s=128 use_text=True use_decoder=True use_video=False \
val_check_interval=0.2 num_workers=16 layer_drop=0.5 use_mpp=False \
load_path=""
