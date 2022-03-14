from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .imagenet_datamodule import ImagenetDataModule
from .imagenet1k_datamodule import Imagenet1kDataModule
from .webvid_datamodule import WebvidDataModule
from .msrvtt_datamodule import MsrvttDataModule
_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "imagenet": ImagenetDataModule,
    "imagenet1k": Imagenet1kDataModule,
    "webvid": WebvidDataModule,
    "msrvtt": MsrvttDataModule
}
