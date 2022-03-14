from model.datasets import WebvidDataset
from .datamodule_base import BaseDataModule


class WebvidDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return WebvidDataset

    @property
    def dataset_cls_no_false(self):
        return WebvidDataset

    @property
    def dataset_name(self):
        return "webvid"
