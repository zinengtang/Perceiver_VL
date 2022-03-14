import functools

from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules
          
class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        
        self.alternate_batch = _config["alternate_batch"]
        
        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        
        for dm in self.dms:
            dm.setup(stage)
            
        if self.alternate_batch:
            
            self.train_dataset = [dm.train_dataset for dm in self.dms]
            self.val_dataset = [dm.val_dataset for dm in self.dms]
            self.test_dataset = [dm.test_dataset for dm in self.dms]
            self.tokenizer = self.dms[0].tokenizer

            self.collate = [functools.partial(
                dm.train_dataset.collate, mlm_collator=dm.mlm_collator,
            ) for dm in self.dms]

            if self.dist:
                self.train_sampler = [DistributedSampler(dset, shuffle=True) for dset in self.train_dataset]
                self.val_sampler = [DistributedSampler(dset, shuffle=False) for dset in self.val_dataset]
                self.test_sampler = [DistributedSampler(dset, shuffle=False) for dset in self.test_dataset]
            else:
                self.train_sampler = None
                self.val_sampler = None
                self.test_sampler = None
        else:
            self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
            self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
            self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
            self.tokenizer = self.dms[0].tokenizer

            self.collate = functools.partial(
                self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
            )

            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.train_sampler = None
                self.val_sampler = None
                self.test_sampler = None
                
    def train_dataloader(self):
        
        if self.alternate_batch:
            loaders = {}
            for i, dset in enumerate(self.train_dataset):
                loaders[str(i)] = DataLoader(
                    self.train_dataset[i],
                    batch_size=self.batch_size,
                    sampler=self.train_sampler[i],
                    num_workers=self.num_workers,
                    collate_fn=self.collate[i],
                )
            
            return loaders

        else:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader

    def val_dataloader(self, batch_size=None):
        
        if self.alternate_batch:
            loaders = {}
            for i, dset in enumerate(self.val_dataset):
                loaders[str(i)] = DataLoader(
                    self.val_dataset[i],
                    batch_size=self.batch_size,
                    sampler=self.val_sampler[i],
                    num_workers=self.num_workers,
                    collate_fn=self.collate[i],
                )
            
            loaders = CombinedLoader(loaders, "max_size_cycle")
            return loaders
        
        else:
            loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size if batch_size is not None else self.batch_size,
                sampler=self.val_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader

    def test_dataloader(self):
        
        if self.alternate_batch:
            loaders = {}
            for i, dset in enumerate(self.test_dataset):
                loaders[str(i)] = DataLoader(
                    self.test_dataset[i],
                    batch_size=self.batch_size,
                    sampler=self.test_sampler[i],
                    num_workers=self.num_workers,
                    collate_fn=self.collate[i],
                )
            
            loaders = CombinedLoader(loaders, "max_size_cycle")
            return loaders
        
        else:
            loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                sampler=self.test_sampler,
                num_workers=self.num_workers,
                collate_fn=self.collate,
            )
            return loader

def joint_collate(self, batch, mlm_collator):
    batch_size = len(batch)
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    
    if 'image' in keys:        
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_sizes = list()

        for img_key in img_keys:
            img = dict_batch[img_key]
            img_sizes += [ii.shape for i in img if i is not None for ii in i]

        for size in img_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(img_keys) != 0:
            max_height = max([i[1] for i in img_sizes])
            max_width = max([i[2] for i in img_sizes])
            
        for img_key in img_keys:
            img = dict_batch[img_key]
            view_size = len(img[0])

            new_images = [
                torch.zeros(batch_size, 3, max_height, max_width)
                for _ in range(view_size)
            ]
            
            for bi in range(batch_size):
                orig_batch = img[bi]
                for vi in range(view_size):
                    if orig_batch is None:
                        new_images[vi][bi] = None
                    else:
                        orig = img[bi][vi]
                        new_images[vi][bi, :, : orig.shape[1], : orig.shape[2]] = orig

            dict_batch[img_key] = new_images

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
    
    elif 'video_data' in keys:

        video_keys = [k for k in list(dict_batch.keys()) if "video" in k]
        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]
            
        for size in video_sizes:
            assert (
                len(size) == 4
            ), f"Collate error, an image should be in shape of (3, H, W), instead of given {size}"

        if len(video_keys) != 0:
            max_video_length = self.max_frames
            max_height = max([i[2] for i in video_sizes])
            max_width = max([i[3] for i in video_sizes])
            
        for video_key in video_keys:
            video = dict_batch[video_key]

            new_videos = torch.zeros(batch_size, max_video_length, 3, max_height, max_width)

            for bi in range(batch_size):
                orig_batch = video[bi]
                
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[video_key] = new_videos

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch     