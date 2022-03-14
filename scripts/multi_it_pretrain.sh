python run.py with data_root='/net/bvisionserver4/playpen10/terran/perceiver/datasets/' gpus=4 num_nodes=1 task_multi_mlm_itm step200k \
per_gpu_batchsize=20 model_type='PerceiverVL_version0' learning_rate=1e-4 \
draw_false_image=1 latent_size_s=128 use_text=True use_decoder=True use_video=False \
val_check_interval=0.2 num_workers=12 layer_drop=0.0 \
load_path="result/multi_mlm_itm_randaug_seed566591265_from_last_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=593.ckpt"