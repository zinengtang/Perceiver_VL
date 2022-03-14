from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .imagenet_datamodule import ImagenetDataModule
from .webvid_datamodule import WebvidDataModule
from .msrvtt_datamodule import MsrvttDataModule

_datamodules = {
    "gcc": ConceptualCaptionDataModule,
    "vqa": VQAv2DataModule,
    "imagenet": ImagenetDataModule,
    "webvid": WebvidDataModule,
    "msrvtt": MsrvttDataModule
}
