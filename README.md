# Perceiver-VL

### **[Perceiver-VL: Efficient Vision-and-Language Modeling with Iterative Latent Attention](https://arxiv.org/abs/2211.11701) [WACV 2023 [bib](https://github.com/zinengtang/Perceiver_VL#citation)]**  
[Zineng Tang*](https://zinengtang.github.io/), [Jaemin Cho*](https://j-min.io/), [Jie Lei](https://jayleicn.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)   

Learning vision-language representation by iterative latent attention that scales with long inputs linearly.

## Introduction
<!-- <p align="center">
  <big><b>Perceiver-VL: Efficient Vision-and-Language Modeling with Iterative Latent Attention (WACV 2023)</b></big>
</p>


<p align="center">
  <big><b>Zineng Tang*, Jaemin Cho*, Jie Lei, Mohit Bansal</b></big>
</p> -->

Perceiver-VL Architecture Overview

<p align="center">
  <img align="middle" width="800" src="assets/architecture.png"/>
</p>


## Install
### Setup `python` environment
```
conda create -n Perceiver-VL python=3.8   # You can also use other environment.
```

### Install other dependencies
```
pip install -r requirements.txt
```


## Training

TODO: Finish datasets/tasks instructions and scripts

### Pretraining (scripts)

```
# Pretrain on Webvid + GCC
bash scripts/co_pretrain.sh
```

```
# Pretrain on Webvid
bash scripts/webvid_pretrain.sh
```

```
# Pretrain on GCC
bash scripts/gcc_pretrain.sh
```

```
# Pretrain on ImageNet
bash scripts/imagenet_pretrain.sh
```

### Pretrained Checkpoint
Download Checkpoint [[link]](https://huggingface.co/murgelab/PerceiverVL/resolve/main/perceivervl_mlm_itm_vtm.ckpt)

### Finetuning on Downstream (scripts)

```
# Fintune on MSRVTT Retrieval
bash scripts/msrvtt_vrtr_finetune.sh
```

```
# Fintune on VQA
bash scripts/vqa_finetune.sh
```


## Code Structure

```
Perceiver_VL
│
├── assets                          # illustrations                          
│   └── architecture.png
│
├── model                           # main source       
│   ├── datamodules                 # pytorch-lightning wrap
│   │   ├── datamodule_base.py
│   │   └── ...          
│   └── datasets                    # Datasets
│   │   ├── vqa_dataset.py     
│   │   └── ...    
│   ├── gadgets     
│   │   └── my_metrics.py           # metric utils
│   ├── modules                     
│   │   ├── heads.py                # model heads
│   │   ├── model_module.py         # pytorch-lightning wrap for model
│   │   ├── model_utils.py          # pytorch-lightning wrap for training metrics
│   │   ├── objectives.py           # pretraining/finetuning objectives
│   │   └── perceiver_vl.py         # main model
│   ├── transforms                  # image transformation utils
│   │   └── ... 
│   └── config.py                   # all configurations
│
├── scripts                         # all scripts
│   ├── vqa_finetune.sh 
│   ├── co_pretrain.sh
│   └── ... 
│
├── run.py                          # main
└── requirements.txt                
```


## Citation
```
@inproceedings{tang2023wacv,
  title     = {Perceiver-VL: Efficient Vision-and-Language Modeling with Iterative Latent Attention},
  author    = {Zineng Tang and Jaemin Cho and Jie Lei and Mohit Bansal},
  booktitle = {WACV},
  year      = {2023}
}
```

## Acknowledgement

Our codebase is based on [ViLT](https://github.com/dandelin/ViLT). 
We thank the authors for their open-source contributions.

## Contact

Zineng Tang (zn.tang.terran@gmail.com)

