import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import model.modules.vision_transformer as vit
import model.modules.perceiver_vl as pvl

import time
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from model.modules import heads, objectives, model_utils
import copy
import warnings
import random
warnings.filterwarnings("ignore")



class PerceiverVL(pl.LightningModule):
    def __init__(self, config, model_type='perceiver'):
        super().__init__()
        self.save_hyperparameters()                
        last_size = config["hidden_size"]
        hs = self.hparams.config["hidden_size"]
        self.learning_rate = config["learning_rate"]   
        
        if  'ViLT' in config['model_type']:
            self.transformer = getattr(vit, self.hparams.config["vit"])(
                    pretrained=config["load_pretrain"], config=self.hparams.config
                )        
            self.model_type = 'transformer'
        elif 'PerceiverVL' in config['model_type']:
            self.transformer = getattr(pvl, self.hparams.config["vit"])(
                    pretrained=config["load_pretrain"], config=self.hparams.config
                )
            self.model_type = 'perceiver'
        else:
            raise NotImplementedError("model class not found")
            
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_layers"],
            num_attention_heads=config["num_heads"],
            intermediate_size=config["hidden_size"] * config["mlp_ratio"],
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_rate"],
            attention_probs_dropout_prob=config["drop_rate"],
        )
        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(objectives.init_weights)
        
        self.token_type_embeddings = nn.Embedding(3, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)       

        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["mlm_video"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0 or config["loss_names"]["vtm"] > 0:
            self.matching_score = heads.ITMHead(config["hidden_size"])
            self.matching_score.apply(objectives.init_weights)

        # ===================== Downstream ===================== #
        if (
            self.hparams.config["load_path"] != ""
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

                    
        if self.hparams.config["loss_names"]["imagenet"] > 0 or self.hparams.config["loss_names"]["imagenet1k"] > 0: 
            vs = self.hparams.config["imagenet_label_size"]
            self.img_classifier = nn.Sequential(
                heads.MeanPooler(last_size),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.img_classifier.apply(objectives.init_weights)
            
        if self.hparams.config["loss_names"]["vqa"] > 0:
            vs = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                heads.MeanPooler(last_size),
                nn.Linear(hs, hs * 2),
                nn.LayerNorm(hs * 2),
                nn.GELU(),
                nn.Linear(hs * 2, vs),
            )
            self.vqa_classifier.apply(objectives.init_weights)

        model_utils.set_metrics(self)
        self.current_tasks = list()
        
        self.apply(objectives.init_weights)
        
        if self.hparams.config["load_path"] != "":
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
                    
        if config['latent_resize']:
            interpolated_pos = F.interpolate(self.transformer.latents.unsqueeze(0).transpose(1,2), size=(config['latent_resize'])).transpose(1,2).squeeze(0)
            self.transformer.latents = nn.Parameter(interpolated_pos)

        
    def infer(
        self,
        batch,
        image_embeds=None,
        image_masks=None,
        video_embeds=None,
        video_masks=None,
        image_token_type_idx=1,
        video_token_type_idx=2,
        mask_text=False,
        mask_visual=False,
    ):
        
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"
            
        videokey = "video_data"
        

        use_image = imgkey in list(batch.keys())
        use_video = videokey in list(batch.keys())
        do_mlm = "_mlm" if mask_text else ""
        
        if f"text_ids{do_mlm}" in list(batch.keys()):
            use_text = True
        else:
            use_text = False
            
        if use_text:    
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            text_embeds = self.text_embeddings(text_ids)
            text_labels_mlm = batch[f"text_labels_mlm"] if f"text_labels_mlm" in batch.keys() and mask_text else None
        else:
            text_ids = None
            text_labels = None
            text_masks = None 
            text_embeds = None
            text_labels_mlm = None

        if use_image:
            if image_embeds is None and image_masks is None:
                img = batch[imgkey][0]
                (
                    image_embeds,
                    image_masks,
                    patch_index,
                    image_labels_mpp,
                    orig_image
                ) = self.transformer.visual_embed(
                    img,
                    max_image_len=self.hparams.config["max_image_len"],
                    mask_it=mask_visual,
                )
            else:
                patch_index = image_labels_mpp = orig_image = None
                
        elif image_embeds is None: 
            image_embeds = None
            image_masks = None
            patch_index = None
            image_labels_mpp = None  
            orig_image = None
        else:
            patch_index = None
            image_labels_mpp = None
            orig_image = None
            
        if use_video:
            if video_embeds is None and video_masks is None:                
                video = batch[videokey]                
                (
                    video_embeds,
                    video_masks,
                    patch_index,
                    video_labels_mpp,
                    orig_video
                ) = self.transformer.visual_embed(
                    video,
                    mask_it=mask_visual,
                )
            else:
                patch_index = video_labels_mpp = orig_video = None
                
                
        elif video_embeds is None:            
            video_embeds = None
            video_masks = None
            patch_index = None
            video_labels_mpp = None
            orig_video = None
        else:
            patch_index = None
            video_labels_mpp = None
            orig_video = None
            
        co_embeds = []
        co_masks = []
                                      
        if use_text:
            text_embeds += self.token_type_embeddings(torch.zeros_like(text_masks))                         
        
        if use_image:
            image_embeds += self.token_type_embeddings(torch.full_like(image_masks, image_token_type_idx))                
            
        if use_video:
            video_embeds += self.token_type_embeddings(torch.full_like(video_masks, video_token_type_idx))
                                
            
        text_feats, image_feats, video_feats = None, None, None
        image_labels_mlm = video_labels_mlm = None

        if self.model_type == 'transformer':
            if use_text:
                co_embeds += [text_embeds]
                co_masks += [text_masks]
            if use_image:
                co_embeds += [image_embeds]
                co_masks += [image_masks]
            if use_video:
                co_embeds += [video_embeds]
                co_masks += [video_masks]
            co_embeds = torch.cat(co_embeds, dim=1)
            co_masks = torch.cat(co_masks, dim=1)   
        
            cls_feats = self.transformer(co_embeds, co_masks)
            if text_embeds is not None:
                text_feats, image_feats = (
                        cls_feats[:, : text_embeds.shape[1]],
                        cls_feats[:, text_embeds.shape[1] :],
                    )
        else:
            cls_feats, text_feats, image_feats, video_feats = self.transformer(text_embeds, text_masks, text_labels_mlm, image_embeds, image_masks, image_labels_mpp, video_embeds, video_masks, video_labels_mpp)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "video_feats": video_feats,
            "cls_feats": cls_feats,
            "video_masks": video_masks,
            "orig_video": orig_video,
            "image_masks": image_masks,
            "orig_image": orig_image,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
            "image_embeds": image_embeds
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Image Text Matching
        if "itm" in self.current_tasks and 'image' in batch.keys():
            ret.update(objectives.compute_itm_wpa(self, batch))
            
        # Video Text Matching
        if "vtm" in self.current_tasks and 'video_data' in batch.keys():
            ret.update(objectives.compute_vtm_wpa(self, batch))
            
        # Masked Language Modeling
        if "mlm" in self.current_tasks and 'image' in batch.keys():
            ret.update(objectives.compute_mlm(self, batch))
            
        if "mlm_video" in self.current_tasks and 'video_data' in batch.keys():
            ret.update(objectives.compute_mlm_video(self, batch))
       
        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))
            
        # Imagenet21k Classification
        if "imagenet" in self.current_tasks:
            ret.update(objectives.compute_imgcls(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))
            
        # Video Retrieval and Text Retrieval
        if "vrtr" in self.current_tasks:
            ret.update(objectives.compute_vrtr(self, batch))

        return ret

    def training_step(self, batch, batch_idx):    
        model_utils.set_task(self)   
        total_loss = 0.0
        if '0'in batch.keys():
            for key in batch:
                output = self(batch[key])
                total_loss += sum([v for k, v in output.items() if "loss" in k])
        else:    
            output = self(batch)
            total_loss += sum([v for k, v in output.items() if "loss" in k])
        return total_loss        
    
    def training_epoch_end(self, outs):
        model_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        model_utils.set_task(self)
        if '0'in batch.keys():
            for batch_key in batch:
                output = self(batch[batch_key])
        else:
            output = self(batch)        
                
    def validation_epoch_end(self, outs):
        model_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        model_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        model_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

