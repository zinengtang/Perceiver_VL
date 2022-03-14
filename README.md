# Perceiver_VL

To speed up the training, we use mixed precision.

## Implementation
```
Run pip install -r requirements.txt
```

## Run Pretraining
```
Bash scripta/webvid_pretrain.sh #For webvid only
Bash scripta/gcc_pretrain.sh #For cc only
Bash scripta/co_pretrain.sh #For both webvid and cc
```


## Run Finetuning
```
Bash scripta/msrvtt_vrtr_finetune.sh #For MSRVTT retrieval finetune
Bash scripta/vqa_finetune.sh #For VQAv2 finetune
```

## Datasets Preparation

### Pretraining
GCC
https://ai.google.com/research/ConceptualCaptions/download

Webvid
https://m-bain.github.io/webvid-dataset/

### Downstream 
VQAv2
https://visualqa.org/download.html



