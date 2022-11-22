python run.py with data_root='datasets/vqav2' gpus=4 num_nodes=1 task_finetune_vqa \
per_gpu_batchsize=64 model_type='PerceiverVL' learning_rate=1e-4 load_pretrain=False latent_size_s=128 use_text=True \
layer_drop=0.5 use_decoder=False num_workers=16 max_epochs=10 \
load_path=""
