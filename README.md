# Perceiver_VL

To speed up the training, we use mixed precision with [Apex](https://github.com/NVIDIA/apex).

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
