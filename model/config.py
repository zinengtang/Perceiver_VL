from sacred import Experiment

ex = Experiment("Perceiver")

def _loss_names(d):
    ret = {
        "itm": 0,
        "vtm": 0,
        "vtm_wpa": 0,
        "mlm": 0,
        "mlm_video":0,
        "mpp": 0,
        "mpp_video": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "vrtr": 0,
        "imagenet": 0,
        "imagenet1k": 0,    
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "perceivervl"
    seed = 0
    datasets = ["gcc"]
    loss_names = _loss_names({"itm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 0
    image_only = False
    imagenet_label_size = 21000
    image_ct_index = [0, 4, 8]

    # Video setting
    video_size = 384
    max_video_len = 20
    draw_false_video = 0
    video_only = False
    max_frames = 8
    use_video = False
    joint_inputs = False
    
    # Text Setting
    vqav2_label_size = 3129    
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0
    use_text = False
    text_ct_index = [1, 5, 9]

    # Transformer Setting
    model_type = 'PerceiverVL'
    vit = "vit_base_patch32_384"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1
    latent_size_s = 128
    latent_resize = False
    use_decoder = False
    layer_drop = 0.5
    use_mpp = False

    # Optimizer Setting
    alternate_batch = False
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.000
    decay_power = 1
    max_epoch = 100
    max_steps = 100000000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_it_recall_metric = False
    get_vt_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    gpus = [0]
    num_nodes = 1
    load_path = ""
    num_workers = 40
    precision = 16
    load_pretrain = False


# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "."
    log_dir = "."
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_cls_imagenet():
    exp_name = "cls_imagenet"
    datasets = ["imagenet"]
    loss_names = _loss_names({"imagenet": 1})
    batch_size = 4096
    imagenet_label_size = 21000
    max_epoch = 100
    max_image_len = 200
    learning_rate = 3e-4
    use_text = False
    
    
@ex.named_config
def task_cls_imagenet1k():
    exp_name = "cls_imagenet1k"
    datasets = ["imagenet1k"]
    loss_names = _loss_names({"imagenet": 1})
    batch_size = 4096
    imagenet_label_size = 1001
    max_epoch = 100
    max_image_len = 200
    learning_rate = 8e-4
    use_text = False
    
# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    max_epoch = 10
    max_image_len = 200
    use_text = True

    
@ex.named_config
def task_multi_mlm_itm():
    exp_name = "multi_mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    max_epoch = 10
    max_image_len = 200
    use_text = True
    

@ex.named_config
def task_mlm_vtm():
    exp_name = "mlm_itm"
    datasets = ["webvid"]
    loss_names = _loss_names({"vtm": 1, "mlm": 1})
    max_epoch = 10
    max_image_len = 200

@ex.named_config
def task_mlm_itm_vtm():
    exp_name = "mlm_itm_vtm"
    datasets = ["gcc", "webvid"]
    loss_names = _loss_names({"mlm": 1, "mlm_video":1, "itm": 1, "vtm": 1})
    max_epoch = 10
    max_image_len = 200
    use_video = True
    get_vt_recall_metric = False
    get_it_recall_metric = False
    max_frames = 14
    draw_false_image = 1
    draw_false_video = 1
    use_text = True
    
@ex.named_config
def task_finetune_msrvtt_vtm():
    exp_name = "finetune_msrvtt_vtm"
    datasets = ["msrvtt"]
    loss_names = _loss_names({"vtm": 1})
    max_frames = 14
    batch_size = 256
    max_epoch = 10
    learning_rate = 1e-4
    val_check_interval = 1.0
    use_video = True
    get_vt_recall_metric = True
    draw_false_text = 16
    draw_false_video = 1

@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
