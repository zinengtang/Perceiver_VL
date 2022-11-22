from glob import glob
from .base_image_dataset import BaseImageDataset
import json
from PIL import Image

class ConceptualCaptionDataset(BaseImageDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split
        
        super().__init__(*args, **kwargs, )
        
    def _load_metadata(self):
        self.folder = self.data_dir+'cc/'
        if self.split == 'train':
            self.metadata = json.load(open(self.folder+'train_dict.json'))
        else:
            self.metadata = json.load(open(self.folder+'valid_dict.json'))
        
    def get_raw_image(self, index):
        return Image.open(self.folder+self.metadata[self.keys[index]]['video_path']).convert("RGB").resize((self.image_size, self.image_size), Image.ANTIALIAS)

    def __getitem__(self, index):
        return self.get_suite(index)
