import random
import torch
import io
import pyarrow as pa
import os
import glob
from tqdm import tqdm
import time
import json 

import numpy as np
from PIL import Image
from model.transforms import keys_to_transforms

# homedir = '/net/bvisionserver4/playpen1/terran/imagenet21k_resized/'
# homedir = '../imagenet21k_resized/'

class ImagenetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=40,
        max_frames=None,
        draw_false_image=0,
        draw_false_text=0,
        image_only=False,
        split='train',
        video_size=None,
        draw_false_video=None,
        video_only=None,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        assert len(transform_keys) >= 1
        super().__init__()
        self.image_paths = []
        self.data_dir = data_dir
        self.classes = json.load(open(data_dir+'classes.json', 'r'))
        
#         if split == 'train':
#             self.image_paths = np.load(data_dir+'train_info.npy')
#             if not os.path.exists(data_dir+'train_classes.npy'):
#                 self.image_classes = [item.split('/')[-1].split('_')[0] for item in tqdm(self.image_paths)]
#                 np.save(data_dir+'train_classes.npy', np.array(self.image_classes))
#             else:
#                 self.image_classes = np.load(data_dir+'train_classes.npy')
            
#         else:
#             self.image_paths = np.load(data_dir+'valid_info.npy')
#             if not os.path.exists(data_dir+'valid_classes.npy'):
#                 self.image_classes = [item.split('/')[-1].split('_')[0] for item in tqdm(self.image_paths)]
#                 np.save(data_dir+'valid_classes.npy', np.array(self.image_classes))
#             else:
#                 self.image_classes = np.load(data_dir+'valid_classes.npy')
#             self.image_classes = [item.split('/')[-1].split('_')[0] for item in tqdm(self.image_paths)]
#             np.save(data_dir+'valid_classes.npy', np.array(self.image_classes))
#         image_paths = os.listdir()
#         if split == 'train':
#             if not os.path.exists(data_dir+'train_info.npy'):
#                 for key in tqdm(self.classes.keys()):
#                     self.image_paths += list(glob.iglob(data_dir+key+'*'))[:-50]                
#                 np.save(data_dir+'train_info.npy', np.array(self.image_paths))
#             else:
#                 self.image_paths = np.load(data_dir+'train_info.npy')
#         else:
#             if not os.path.exists(data_dir+'valid_info.npy'):
#                 for key in tqdm(self.classes.keys()):
#                     self.image_paths += list(glob.iglob(data_dir+key+'*'))[-50:]
#                 np.save(data_dir+'valid_info.npy', np.array(self.image_paths))
#             else:
#                 self.image_paths = np.load(data_dir+'valid_info.npy')
        
        if split == 'train':
            if not os.path.exists(data_dir+'train_info.npy'):
                for directory in tqdm(list(glob.iglob(data_dir+'imagenet21k_train/*')) + list(glob.iglob(data_dir+'imagenet21k_small_classes/*'))):                
                    self.image_paths += list(glob.iglob(directory+'/*'))[:-10]
                np.save(data_dir+'train_info.npy', np.array(self.image_paths))
            else:
                self.image_paths = np.load(data_dir+'train_info.npy')
            if not os.path.exists(data_dir+'train_classes.npy'):
                self.image_classes = [item.split('/')[-1].split('_')[0] for item in tqdm(self.image_paths)]
                np.save(data_dir+'train_classes.npy', np.array(self.image_classes))
            else:
                self.image_classes = np.load(data_dir+'train_classes.npy')
            
        else:
            if not os.path.exists(data_dir+'valid_info.npy'):
                for directory in tqdm(list(glob.iglob(data_dir+'imagenet21k_train/*')) + list(glob.iglob(data_dir+'imagenet21k_small_classes/*'))):                
                    self.image_paths += list(glob.iglob(directory+'/*'))[-10:]
                np.save(data_dir+'valid_info.npy', np.array(self.image_paths))
            else:
                self.image_paths = np.load(data_dir+'valid_info.npy')
            if not os.path.exists(data_dir+'valid_classes.npy'):
                self.image_classes = [item.split('/')[-1].split('_')[0] for item in tqdm(self.image_paths)]
                np.save(data_dir+'valid_classes.npy', np.array(self.image_classes))
            else:
                self.image_classes = np.load(data_dir+'valid_classes.npy')
        print(len(self.image_paths))
#         self.classes = os.listdir(data_dir+'imagenet21k_train/') + os.listdir(data_dir+'imagenet21k_small_classes/')
#         self.classes = dict(zip(self.classes, range(len(self.classes))))
        if self.data_dir == "datasets/winter21_whole/":
            self.data_dir = ""
        elif "/net/bvisionserver14" in self.data_dir:
            self.data_dir = "/net/bvisionserver14/playpen2/terran/perceiver/"
        self.draw_false_image = draw_false_image
        self.image_only = image_only
        self.image_size = image_size
        self.transforms = keys_to_transforms(transform_keys, size=self.image_size)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.image_paths)


    def get_image(self, index,):
#         start=time.time()
#         '/net/bvisionserver14/playpen2/terran/perceiver/'+

#         image = np.array(Image.open(self.image_paths[index]).convert("RGB").resize((self.image_size, self.image_size),Image.ANTIALIAS))
        image_path = self.data_dir+self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        
#     .resize((self.image_size, self.image_size),Image.ANTIALIAS)
#     .convert("RGB")
#         print(image.mode)
#         if not image.mode == 'RGB':
# #             print(image.mode)
#             image = image.convert("RGB")
# #             image.save(self.image_paths[index])

        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size),Image.ANTIALIAS)
#             image.save(self.image_paths[index])
#         print(time.time()-start)    
        image_tensor = [tr(image) for tr in self.transforms] 
        return {
            "image": image_tensor,
            "raw_index": index,
        }

    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.image_paths) - 1)
        image = Image.open(self.image_paths[index])
#         .convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size),Image.ANTIALIAS)
            image.save(self.image_paths[index])
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}
    
    def get_class(self, index,):
        item_class = self.image_classes[index]
#         self.image_paths[index].split('/')[-1].split('_')[0]
        item_class = self.classes[item_class]
        return {
            "label": item_class,
            "raw_index": index,
        }

    def get_suite(self, index):        
#         result = None
#         while result is None:
#             try:
        ret = dict()
        ret.update(self.get_image(index))
        
        if not self.image_only:
            cls = self.get_class(index)
            ret.update(cls)
#         print('2', time.time()-start)
        for i in range(self.draw_false_image):
            ret.update(self.get_false_image(i))
#         print('3', time.time()-start)
#         result = True
#             except Exception as e:
#                 print(f"Error while read sample idx {index}")
#                 index = random.randint(0, len(self.index_mapper) - 1)
        return ret
    
    def __getitem__(self, index):
        return self.get_suite(index)
    
    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

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
