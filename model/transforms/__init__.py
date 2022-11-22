from .pixelbert import (
    pixelbert_transform,
    pixelbert_transform_randaug,
)

_transforms = {
    "pixelbert": pixelbert_transform,
    "pixelbert_randaug": pixelbert_transform_randaug,
}


def keys_to_transforms(keys: list, size=384):
    return [_transforms[key](size) for key in keys]
