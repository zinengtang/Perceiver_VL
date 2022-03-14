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

        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__(*args, **kwargs,)        

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
   
    def _load_metadata(self):
        if not os.path.exists('datasets/webvid/train_dict.json'):
            train=list(glob.iglob('/playpen1/terran/webvid/train_images/*'))+\
                list(glob.iglob('/playpen2/terran/webvid/train_images/*'))+\
                list(glob.iglob('/playpen-ssd/terran/webvid/train_images/*'))

            valid=list(glob.iglob('/playpen2/terran/webvid/valid_images/*'))

            train_csv= pd.read_csv('datasets/webvid/results_2M_train.csv')
            valid_csv= pd.read_csv('datasets/webvid/results_2M_val.csv')


            train_dict = dict(zip(map(str, list(train_csv['videoid'])), list(train_csv['name'])))

            for path in tqdm(train):
                key = path.split('/')[-1].split('.')[0]
                train_dict[key] = {'video_path': path, 'text': train_dict[key]}

            valid_dict = dict(zip(map(str, list(valid_csv['videoid'])), list(valid_csv['name'])))

            for path in tqdm(valid):
                key = path.split('/')[-1].split('.')[0]
                valid_dict[key] = {'video_path': path, 'text': valid_dict[key]}

            json.dump(train_dict, open('datasets/webvid/train_dict.json', 'w'), )
            json.dump(valid_dict, open('datasets/webvid/valid_dict.json', 'w'), )

        if self.split=='train':
            self.metadata = json.load(open('datasets/webvid/train_dict.json'))
        else:
            self.metadata = json.load(open('datasets/webvid/valid_dict.json'))
            self.metadata = dict(list(self.metadata.items())[:2048])
