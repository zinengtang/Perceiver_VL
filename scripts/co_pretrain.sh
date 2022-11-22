python run.py with data_root='datasets/cc/' gpus=4 num_nodes=1 task_mlm_itm_vtm step200k \
per_gpu_batchsize=3 model_type='PerceiverVL' learning_rate=1e-5 \
draw_false_video=1 draw_false_image=1 max_frames=8 latent_size_s=128 use_text=True use_video=True alternate_batch=True use_decoder=True \
val_check_interval=0.05 num_workers=10 layer_drop=0.0 \
load_path=""
