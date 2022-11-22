from glob import glob
import json
from PIL import Image

from .base_image_dataset import BaseImageDataset

class ConceptualCaptionDataset(BaseImageDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split
        
        super().__init__(*args, **kwargs, )
        
    def _load_metadata(self):
        self.data_dir = self.data_dir+'cc/'
        if self.split == 'train':
            self.metadata = json.load(open(self.data_dir+'train_dict.json'))
        else:
            self.metadata = json.load(open(self.data_dir+'valid_dict.json'))
        
    def get_raw_image(self, index):
        return Image.open(os.path.join(self.folder, self.metadata[self.keys[index]]['video_path'])).convert("RGB").resize((self.image_size, self.image_size), Image.ANTIALIAS)

    def __getitem__(self, index):
        return self.get_suite(index)
