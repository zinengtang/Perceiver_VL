python run.py with data_root='datasets/cc/' gpus=4 num_nodes=1 task_mlm_itm step200k \
per_gpu_batchsize=16 model_type='PerceiverVL' learning_rate=3e-4 \
draw_false_image=1 latent_size_s=128 use_text=True use_decoder=True use_video=False \
val_check_interval=0.2 num_workers=12 layer_drop=0.5 \
load_path=""
