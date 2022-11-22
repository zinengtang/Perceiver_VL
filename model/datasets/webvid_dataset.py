import os
import glob
from tqdm import tqdm
import json

from .base_video_dataset import BaseVideoDataset

class WebvidDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split
        super().__init__(*args, **kwargs,)        

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
   
    def _load_metadata(self):
        self.data_dir = os.path.join(self.data_dir, 'webvid/')
        if self.split=='train':
            self.metadata = json.load(open(os.path.join(self.data_dir, 'train_dict.json')))
        else:
            self.metadata = json.load(open(os.path.join(self.data_dir, 'valid_dict.json')))
            self.metadata = dict(list(self.metadata.items()))
