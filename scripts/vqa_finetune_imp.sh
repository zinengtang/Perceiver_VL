python run.py with data_root='datasets/vqav2_bin' gpus=4 num_nodes=1 task_finetune_vqa \
per_gpu_batchsize=64 model_type='PerceiverVL_version0' learning_rate=1e-4 load_pretrain=False latent_size_s=128 use_text=True \
layer_drop=1.0 use_decoder=False num_workers=16 max_epochs=10 \
load_path="/net/bvisionserver14/playpen2/terran/perceiver/result/cls_imagenet_seed634686292_from_last_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=158.ckpt"
# load_path="result/version_103/checkpoints/last.ckpt"

# python run.py with data_root='datasets/vqav2_bin' gpus=4 num_nodes=1 task_finetune_vqa \
# per_gpu_batchsize=16 model_type='ViLT' learning_rate=1e-4 load_pretrain=True use_text=True \
# layer_drop=1.0 use_decoder=False num_workers=16 \
# load_path="result/finetune_vqa_seed249167836_from__ViLT/version_0/checkpoints/epoch=9-step=22740.ckpt"
# load_path="/net/bvisionserver4/playpen10/terran/perceiver/result/mlm_itm_mpp_randaug_seed0_from_last_PerceiverVL_version0/version_103/checkpoints/last.ckpt"


# latent_resize=32
# "result/finetune_vqa_randaug_seed344598978_from_epoch=14-step=35485_PerceiverVL_version0/version_0/checkpoints/epoch=0-step=249.ckpt"