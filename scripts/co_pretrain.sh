python run.py with data_root='/playpen/terran/cc/' gpus=3 num_nodes=1 task_mlm_itm_vtm step200k \
per_gpu_batchsize=3 model_type='PerceiverVL_version0' learning_rate=1e-5 \
draw_false_video=1 draw_false_image=1 max_frames=1 latent_size_s=128 use_text=True use_video=True alternate_batch=True use_mpp=False use_decoder=True \
val_check_interval=0.05 num_workers=10 layer_drop=0.0 \
load_path="result/mlm_itm_vtm_seed593436609_from_epoch=0-step=145_PerceiverVL_version0/version_0/checkpoints/last.ckpt"



# mlm vtm itm
# load_path="result/mlm_itm_vtm_seed70212416_from_epoch=0-step=71_PerceiverVL_version0/version_0/checkpoints/last.ckpt"

# vilt
# "result/mlm_itm_vtm_seed843401919_from_epoch=0-step=35_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=179.ckpt"

# ori load_path="result/mlm_itm_vtm_seed725777537_from_last_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=35.ckpt"

# load to perceive v1
# load_path="result/mlm_itm_vtm_seed25870576_from_last_PerceiverVL_version0/version_0/checkpoints/epoch=1-step=933.ckpt"

# decoder pos embed
# result/mlm_itm_vtm_seed462989820_from_epoch=0-step=179_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=717.ckpt
