import random
import torch
import io
import pyarrow as pa
import os
import glob
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
from model.datasets.rawvideo_utils import RawVideoExtractorCV2
from .base_video_dataset import BaseVideoDataset


class MsrvttDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split
        self.cut = "full-val"
        
        super().__init__(*args, **kwargs,)
        
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
    
    def _get_raw_text(self, index, sample_type="rand"):
        text = self.metadata[self.keys[index]]['text']

        if sample_type == "rand":
            text = random.choice(text)
        else:
            text = text[0]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return text, encoding

    def _load_metadata(self):
        json_fp = os.path.join(self.metadata_dir, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(self.metadata_dir, 'high-quality', 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
#         if self.cut == "miech":
#             train_list_path = "train_list_miech.txt"
#             test_list_path = "test_list_miech.txt"
#         elif self.cut == "jsfusion":
#             train_list_path = "train_list_jsfusion.txt"
#             test_list_path = "val_list_jsfusion.txt"
#             js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
#         elif self.cut in {"full-val", "full-test"}:
        train_list_path = "train_list_full.txt"
        if self.cut == "full-val":
            test_list_path = "val_list_full.txt"
        else:
            test_list_path = "test_list_full.txt"
#         elif self.cut in challenge_splits:
#             train_list_path = "train_list.txt"
#             if self.cut == "val":
#                 test_list_path = f"{self.cut}_list.txt"
#             else:
#                 test_list_path = f"{self.cut}.txt"
#         else:
#             msg = "unrecognised MSRVTT split: {}"
#             raise ValueError(msg.format(self.cut))

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])
        self.split_sizes = {'train': len(train_df), 'val': len(test_df), 'test': len(test_df)}

        if self.split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        self.metadata = df.groupby(['image_id'])['caption'].apply(list)
        if self.subsample < 1:
            self.metadata = self.metadata.sample(frac=self.subsample)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and self.split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            self.metadata = new_res['test_caps']
            
        tempdata = {}
        self.metadata=dict(self.metadata)
        for key in tqdm(self.metadata):
            tempdata[key] = {'video_path': 'datasets/msrvtt/videos/all/'+key+'.mp4', 'text': self.metadata[key]}
        self.metadata = tempdata
        