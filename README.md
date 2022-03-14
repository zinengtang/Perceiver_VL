# Perceiver_VL

## Installation
```Bash
pip install -r requirements.txt
```

## Pretraining
```bash
bash scripta/webvid_pretrain.sh # Pretraining on WebVid
bash scripta/gcc_pretrain.sh # Pretrainig on CC
bash scripta/co_pretrain.sh # Pretraining on CC+WebVid
```


## Finetuning
```bash
bash scripts/msrvtt_vrtr_finetune.sh # Finetuning on MSRVTT
bash scripts/vqa_finetune.sh # Finetuning on VQAv2
```
