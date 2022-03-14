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

class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir = '.',
        transform_keys=None,
        image_size=None,
        video_size=224,
        max_text_len=40,
        max_frames=8,
        draw_false_image=None,
        draw_false_video=1,
        draw_false_text=0,
        image_only=None,
        video_only=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()
              
        self.metadata_dir = data_dir
        
        self.video_loader = RawVideoExtractorCV2(size=video_size, max_frames=max_frames)

        self.draw_false_video = draw_false_video
        self.video_only = video_only
        self.video_size = video_size
        self.video_only = video_only
        
        self.draw_false_text = draw_false_text 
        self.draw_false_video = draw_false_video
        self.max_frames = max_frames
        self.max_text_len = max_text_len
#         self.transforms = video_transform_dict()
#         self.split = split
        self.subsample = 1
        
        self._load_metadata()
        self.keys = list(self.metadata.keys())
               
        self.index_mapper = list(range(len(self.keys)))
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
        
#     def _get_video_path(self, index):
#         return os.path.join(self.metadata_dir, 'videos', 'all', self.keys[index] + '.mp4')

    def _get_video(self, index):
        video_data = self.video_loader.get_video_data(self.metadata[self.keys[index]]['video_path'])
#         print(video_data[0])
        return {"video_data": video_data,
                "raw_index": index,
                "v_index": index,}   
    
    def _get_false_video(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        video_data = self.video_loader.get_video_data(self.metadata[self.keys[random_index]]['video_path'])
        return {f"false_video_{rep}": video_data}
    
    def _get_raw_text(self, index, sample_type="rand"):
        text = self.metadata[self.keys[index]]['text']

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return text, encoding
    
    def _get_text(self, index, ):
        text, encoding = self._get_raw_text(index)
        return {"text": (text, encoding),
                "raw_index": index,}    

    def _get_false_text(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        text, encoding = self._get_raw_text(random_index)
        return {f"false_text_{rep}": (text, encoding)}
    
    def get_suite(self, index):
        result = None
        while result is None:
#             try:
                ret = dict()
                ret.update(self._get_video(index))
                if not self.video_only:
                    ret.update(self._get_text(index))                    
                result = True
                for i in range(self.draw_false_text):
                    ret.update(self._get_false_text(i))
                for i in range(self.draw_false_video):
                    ret.update(self._get_false_video(i))
#             except Exception as e:
#                 print(f"Error while read sample idx {index}")
#                 index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    
    def __getitem__(self, index):
        return self.get_suite(index)
    
    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        video_keys = [k for k in list(dict_batch.keys()) if "video" in k]
        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]
            
        for size in video_sizes:
            assert (
                len(size) == 4
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(video_keys) != 0:
            max_video_length = self.max_frames
            max_height = max([i[2] for i in video_sizes])
            max_width = max([i[3] for i in video_sizes])
            
        for video_key in video_keys:
            video = dict_batch[video_key]

            new_videos = torch.zeros(batch_size, max_video_length, 3, max_height, max_width)

            for bi in range(batch_size):
                orig_batch = video[bi]
                
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
#                     print(orig.shape)
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
#             print(new_videos.size())
            dict_batch[video_key] = new_videos

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
