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

        if self.split=='train':
            self.metadata = json.load(open('datasets/webvid/train_dict.json'))
        else:
            self.metadata = json.load(open('datasets/webvid/valid_dict.json'))
            self.metadata = dict(list(self.metadata.items())[:2048])
