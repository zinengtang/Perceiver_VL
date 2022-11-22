from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment


def pixelbert_transform(img_size=384):
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug():
    trs = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs
