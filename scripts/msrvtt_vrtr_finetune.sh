python run.py with data_root='datasets/msrvtt/' gpus=4 num_nodes=1 task_finetune_msrvtt_vtm step200k \
per_gpu_batchsize=8 model_type='PerceiverVL' learning_rate=1e-5 \
draw_false_video=1 draw_false_text=1 max_frames=8 latent_size_s=128 use_text=True use_video=True layer_drop=0.5 \
val_check_interval=1.0 num_workers=16 get_vt_recall_metric=True use_decoder=True \
load_path=""
