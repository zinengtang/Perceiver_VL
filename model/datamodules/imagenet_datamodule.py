from model.datasets import ImagenetDataset
from .datamodule_base import BaseDataModule


class ImagenetDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ImagenetDataset

    @property
    def dataset_cls_no_false(self):
        return ImagenetDataset

    @property
    def dataset_name(self):
        return "imagenet"
