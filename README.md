# Perceiver_VL

## Installation
```Bash
pip install -r requirements.txt
```

## Pretraining
```bash
bash scripts/webvid_pretrain.sh # Pretraining on WebVid
bash scripts/gcc_pretrain.sh # Pretrainig on CC
bash scripts/co_pretrain.sh # Pretraining on CC+WebVid
```


## Finetuning
```bash
bash scripts/msrvtt_vrtr_finetune.sh # Finetuning on MSRVTT
bash scripts/vqa_finetune.sh # Finetuning on VQAv2
```
