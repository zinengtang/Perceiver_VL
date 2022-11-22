python run.py with data_root='datasets/' gpus=4 num_nodes=1 task_mlm_vtm_mpp step200k \
per_gpu_batchsize=16 model_type='PerceiverVL' learning_rate=1e-5 \
draw_false_video=1 draw_false_image=1 max_frames=8 latent_size_s=128 latent_size_t=1 use_text=True use_decoder=True use_video=True \
alternate_batch=False layer_drop=0.5 val_check_interval=0.2 num_workers=12 \
