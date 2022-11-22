
import math
import logging
from functools import partial
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
import urllib
import warnings

from model.modules import objectives

from functools import partial
from tqdm import tqdm

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import StdConv2dSame, DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.resnetv2 import ResNetV2
from timm.models.registry import register_model
from torchvision import transforms

_logger = logging.getLogger(__name__)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


inception_unnormalize = transforms.Compose(
    [UnNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }



class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


    
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
            
    def forward(self, x, context=None, mask=None):       
        skip_x = x
        B, N, C = x.shape
        q = self.q(x).reshape(B, x.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, context.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, context.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn = torch.tanh(attn)
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, skip_x, attn 
        
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):            
        B, N, C = x.shape          
        
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_context=False,
        post_norm=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_context:
            self.attn = CrossAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )            
        else:           
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
        self.post_norm = post_norm
        if post_norm:
            self.post_norm_layer = norm_layer(dim)
        
    def forward(self, x, context=None, mask=None):
        
        if context is None:
            _x, attn = self.attn(self.norm1(x), mask=mask)
        else:
            _x, x, attn = self.attn(self.norm1(x), context=context, mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.post_norm:
            self.post_norm_layer(x)
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x

import numpy as np

class PerceiverVL(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        add_norm_before_transformer=False,
        no_patch_embed_bias=False,
        config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        drop_rate = drop_rate if config is None else config["drop_rate"]

        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.add_norm_before_transformer = add_norm_before_transformer

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.patch_size = patch_size
        self.patch_dim = img_size // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cls_token_decoder = nn.Parameter(torch.zeros(1, 64, embed_dim))
        
        if add_norm_before_transformer:
            self.pre_norm = norm_layer(embed_dim)
            
        if config["use_video"]:
            self.temporal_embed = nn.Parameter(torch.zeros(1, 64, config["hidden_size"]))
            self.video_dim = [config["max_frames"], (config["video_size"]//config["patch_size"])**2+1]
            self.latent_size = config["latent_size_s"]
            self.joint_inputs = True
            self.max_frames = config["max_frames"]
            
        else:            
            self.latent_size = config["latent_size_s"]
            self.joint_inputs = False
            
        self.depth = depth
        self.cross_layers_visual = [0, 4, 8]        
        num_cross_blocks = len(self.cross_layers_visual)
        
        self.crossatt_blocks_visual = nn.ModuleList(
            [                                         
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    use_context=True,
                )               
                for i in range(num_cross_blocks)
            ]
        )
        
        self.use_mpp = config['use_mpp']
        if config['use_mpp'] and config["use_decoder"]:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.mpp_pos = nn.Embedding(num_patches+1, embed_dim)
            self.mpp_pos_t = nn.Embedding(64, embed_dim)
            self.mpp_type = nn.Embedding(2, embed_dim)
                
        if config['use_text'] and config["use_decoder"]:
            num_decode_blocks = 0
            
            self.mlm_pos = nn.Embedding(config["max_text_len"], embed_dim)
            self.mlm_type = nn.Embedding(2, embed_dim)                
            self.decoder_block_text = nn.ModuleList(
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        use_context=True,
                        post_norm=False
                    )
                    for i in range(1)
                ] +
                [
                    Block(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        use_context=False,
                        post_norm=(i==num_decode_blocks-1)
                    )
                    for i in range(num_decode_blocks)
                ]
            )
        self.use_decoder = config["use_decoder"]
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        
        self.co_norm = norm_layer(embed_dim)        
        self.norm = norm_layer(embed_dim)
        self.decoder_norm = norm_layer(embed_dim)
        
        self.layer_drop = config["layer_drop"]
        
        
        self.latents = nn.Parameter(torch.randn(config['latent_size_s'], embed_dim))
        
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def mask_tokens(self, orig, feats):
        indices_replaced = (
            torch.bernoulli(torch.full(feats.shape[:-1], 0.75)).bool()
        )
        feats[indices_replaced] = self.mask_token.to(feats)
        
        B, C, H, W = orig.size()
        B, p, h = feats.size()
        indices_replaced_img = indices_replaced.reshape(B, int(math.sqrt(p)), int(math.sqrt(p)))
        indices_replaced_img = torch.repeat_interleave(indices_replaced_img, int(H//math.sqrt(p)), dim=1)
        indices_replaced_img = torch.repeat_interleave(indices_replaced_img, int(W//math.sqrt(p)), dim=2)
        indices_replaced_img = indices_replaced_img.unsqueeze(1).repeat(1, C, 1, 1)
        orig[indices_replaced_img] = -100.0

        return feats, indices_replaced, orig

    def visual_embed(self, _x, max_image_len=200, mask_it=False):
        
        if len(_x.size()) == 5:
            use_video = True
            ori_shape = _x.size()
            video_mask = _x.reshape(ori_shape[0], ori_shape[1], -1).mean(-1)==0
            _x = _x.reshape(ori_shape[0]*ori_shape[1], ori_shape[2], ori_shape[3], ori_shape[4])
        else:
            use_video = False
        _, _, ph, pw = self.patch_embed.proj.weight.shape

        x = self.patch_embed(_x)        
        x_mask = torch.ones_like(_x.sum(dim=1) != 0).float()[:, None, :, :]
        
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]
        B, C, H, W = x.shape
        spatial_pos = (
            self.pos_embed[:, 1:, :]
            .transpose(1, 2)
            .view(1, C, self.patch_dim, self.patch_dim)
        )
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos, size=(h, w), mode="bilinear", align_corners=True,
                    ),
                    (0, W - w, 0, H - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1])
                ),
                dim=-1,
            )[None, None, :, :, :]
            .expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
            .flatten(1, 3)
        )
        x_mask = x_mask.flatten(1)

        if mask_it:
            x, label, orig = self.mask_tokens(_x, x)

        if (
            max_image_len < 0
            or max_image_len is None
            or not isinstance(max_image_len, int)
        ):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            eff = x_h * x_w
            max_image_len = eff.max()
        else:
            eff = x_h * x_w
            max_image_len = min(eff.max(), max_image_len)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [
            non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows
        ]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_len - v for v in valid_nums]

        select = list()
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_len)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(
                    torch.ones(nv).float(), p, replacement=True
                )
                select.append(
                    torch.cat(
                        [valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0,
                    )
                )

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(B, -1, C)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(B, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(B, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(B, -1, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.pos_embed[:, 0, :][:, None, :].expand(B, -1, -1), pos_embed), dim=1
        )
        
        x = x + pos_embed
        if use_video:
            x_mask = x_mask.view(ori_shape[0], ori_shape[1], x_mask.size(1))
            x_mask= torch.cat([torch.ones(ori_shape[0], ori_shape[1], 1).to(x_mask), x_mask], dim=-1)
            x_mask[video_mask]=0
            x_mask = x_mask.view(ori_shape[0], -1)
            
            x = x.view(ori_shape[0], ori_shape[1] * x.size(1), x.size(-1))
            x += torch.repeat_interleave(self.temporal_embed[:, :self.max_frames], x.size(1)//ori_shape[1], dim=1)
            
        else:
            x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        x = self.pos_drop(x)

        if self.add_norm_before_transformer:
            x = self.pre_norm(x)

        if mask_it:
            return x, x_mask, (patch_index, (H, W)), label, orig
        else:
            return x, x_mask, (patch_index, (H, W)), None, None
        
    def forward_unimo(self, inputs):
        
        x = self.latents.unsqueeze(0).repeat(inputs.size(0), 1, 1)
        for i in range(self.depth):                    
            if i in self.cross_layers_visual:
                if i in [0] or random.random() <= self.layer_drop:
                    x, _ = self.crossatt_blocks_visual[self.cross_layers_visual.index(i)](x, inputs, masks)

            x, _ = self.blocks[i](x)
        
        return x
    
    def forward_single(self, text_embeds=None, text_masks=None, text_labels_mlm=None, image_embeds=None, image_masks=None, image_labels_mpp=None, video_embeds=None, video_masks=None, video_labels_mpp=None):
                        
        if image_embeds is not None:
            inputs = self.co_norm(image_embeds)
            masks = image_masks
            
        if video_embeds is not None:
            inputs = self.co_norm(video_embeds)
            masks = video_masks
            
        if text_embeds is not None:
            text_inputs = self.co_norm(text_embeds)        
            inputs = torch.cat([inputs, text_inputs], 1)
            masks = torch.cat([masks, text_masks], 1)
            
        if False:
            x = inputs
            for i in range(self.depth):                    
                x, _ = self.blocks[i](x, mask=masks)
        else:
            x = self.latents.unsqueeze(0).repeat(inputs.size(0), 1, 1)
            for i in range(self.depth):                    
                if i in self.cross_layers_visual:
                    if i in [0] or random.random() <= self.layer_drop:
                        x, _ = self.crossatt_blocks_visual[self.cross_layers_visual.index(i)](x, inputs, masks)

                x, _ = self.blocks[i](x)

        x = self.norm(x)
        
        text_feats = image_feats = video_feats = None
        if self.use_decoder:
                                    
            decoder_mask = torch.ones([1, 1]).repeat(x.size(0), 1).to(x.device)
            decoder_feats = self.cls_token_decoder[:, :1].repeat(x.size(0), 1, 1)                           
            visual_output_length = 0                                                
                    
            if text_labels_mlm is not None:
                
                position_ids = torch.arange(text_embeds.size(1)).expand((1, -1)).to(text_embeds.device)
                position_encodings = self.mlm_pos(position_ids)                
                type_ids = (text_labels_mlm==-100).to(text_embeds.device).long()
                tpye_encodings = self.mlm_type(type_ids)
                
                decoder_text_pos = position_encodings + tpye_encodings
                
                decoder_feats = torch.cat([decoder_feats, decoder_text_pos], 1)
                decoder_mask = torch.cat([decoder_mask, text_masks], 1)
                
            for i in range(len(self.decoder_block_text)):
                if i == 0:
                    decoder_feats, _ = self.decoder_block_text[i](decoder_feats, x)
                else:
                    decoder_feats, _ = self.decoder_block_text[i](decoder_feats, mask=decoder_mask)
            decoder_feats = self.decoder_norm(decoder_feats)                    
            if image_embeds is not None:
                image_feats = decoder_feats[:, 1:visual_output_length+1]
            if video_embeds is not None:
                video_feats = decoder_feats[:, 1:visual_output_length+1]            
            if text_embeds is not None:           
                text_feats = decoder_feats[:, 1+visual_output_length:]
                
        return decoder_feats, text_feats, image_feats, video_feats
    
    def forward_multi(self, text_embeds=None, text_masks=None, text_labels_mlm=None, image_embeds=None, image_masks=None, image_labels_mpp=None, video_embeds=None, video_masks=None, video_labels_mpp=None):
                        
        if image_embeds is not None:
            inputs = self.co_norm(image_embeds)
            masks = image_masks
            x = self.forward_unimo(inputs)
            image_feats = x
            
        if video_embeds is not None:
            inputs = self.co_norm(video_embeds)
            masks = video_masks
            x = self.forward_unimo(inputs)
            video_feats = x
            
        if text_embeds is not None:
            text_inputs = self.co_norm(text_embeds)        
            inputs = torch.cat([inputs, text_inputs], 1)           
            masks = torch.cat([masks, text_masks], 1)
            x = self.forward_unimo(inputs)
            text_feats = x
            x = torch.cat([x, text_feats], 1)
                   
        decoder_feats = self.norm(x)
                
        return decoder_feats, text_feats, image_feats, video_feats
    
    def forward_mixed(self, text_embeds=None, text_masks=None, text_labels_mlm=None, image_embeds=None, image_masks=None, image_labels_mpp=None, video_embeds=None, video_masks=None, video_labels_mpp=None):
                        
        if image_embeds is not None:
            inputs = self.co_norm(image_embeds)
            masks = image_masks
            x = self.forward_unimo(inputs)
            
        if video_embeds is not None:
            inputs = self.co_norm(video_embeds)
            masks = video_masks
            x = self.forward_unimo(inputs)
            
        if text_embeds is not None:
            text_inputs = self.co_norm(text_embeds)        
            inputs = torch.cat([inputs, text_inputs], 1)
            x = torch.cat([x, x_text], 1)
            masks = torch.cat([masks, text_masks], 1)                   
            
        x = self.latents.unsqueeze(0).repeat(inputs.size(0), 1, 1)
        for i in range(self.depth):                    
            x, _ = self.blocks[i](x)

        x = self.norm(x)
        
        text_feats = image_feats = video_feats = None
        if self.use_decoder:
                                    
            decoder_mask = torch.ones([1, 1]).repeat(x.size(0), 1).to(x.device)
            decoder_feats = self.cls_token_decoder[:, :1].repeat(x.size(0), 1, 1)                           
            visual_output_length = 0                                                
                    
            if text_labels_mlm is not None:
                
                position_ids = torch.arange(text_embeds.size(1)).expand((1, -1)).to(text_embeds.device)
                position_encodings = self.mlm_pos(position_ids)                
                type_ids = (text_labels_mlm==-100).to(text_embeds.device).long()
                tpye_encodings = self.mlm_type(type_ids)
                
                decoder_text_pos = position_encodings + tpye_encodings
                
                decoder_feats = torch.cat([decoder_feats, decoder_text_pos], 1)
                decoder_mask = torch.cat([decoder_mask, text_masks], 1)
                
            for i in range(len(self.decoder_block_text)):
                if i == 0:
                    decoder_feats, _ = self.decoder_block_text[i](decoder_feats, x)
                else:
                    decoder_feats, _ = self.decoder_block_text[i](decoder_feats, mask=decoder_mask)
                    
            decoder_feats = self.decoder_norm(decoder_feats)                    
            if image_embeds is not None:
                image_feats = decoder_feats[:, 1:visual_output_length+1]
            if video_embeds is not None:
                video_feats = decoder_feats[:, 1:visual_output_length+1]            
            if text_embeds is not None:           
                text_feats = decoder_feats[:, 1+visual_output_length:]
                
        return decoder_feats, text_feats, image_feats, video_feats
    
    def forward(self, text_embeds=None, text_masks=None, text_labels_mlm=None, image_embeds=None, image_masks=None, image_labels_mpp=None, video_embeds=None, video_masks=None, video_labels_mpp=None):
        
        if config['architecture'] == 'single':
            return self.forward_single(text_embeds=text_embeds, text_masks=text_masks, text_labels_mlm=text_labels_mlm, image_embeds=image_embeds, image_masks=image_masks, image_labels_mpp=image_labels_mpp, video_embeds=video_embeds, video_masks=video_masks, video_labels_mpp=video_labels_mpp)
        elif config['architecture'] == 'multi':
            return self.forward_multi(text_embeds=text_embeds, text_masks=text_masks, text_labels_mlm=text_labels_mlm, image_embeds=image_embeds, image_masks=image_masks, image_labels_mpp=image_labels_mpp, video_embeds=video_embeds, video_masks=video_masks, video_labels_mpp=video_labels_mpp)
        elif config['architecture'] == 'mixed':
            return self.forward_mixed(text_embeds=text_embeds, text_masks=text_masks, text_labels_mlm=text_labels_mlm, image_embeds=image_embeds, image_masks=image_masks, image_labels_mpp=image_labels_mpp, video_embeds=video_embeds, video_masks=video_masks, video_labels_mpp=video_labels_mpp)
        else:
            raise

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info("Position embedding grid-size from %s to %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def _create_vision_transformer(variant, pretrained=False, distilled=False, **kwargs):
    default_cfg = default_cfgs[variant]
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model_cls = PerceiverVL
    model = model_cls(
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        **kwargs,
    )
    model.default_cfg = default_cfg

    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes,
            in_chans=kwargs.get("in_chans", 3),
            strict=False,
        )
    else:
        model.apply(objectives.init_weights)
    return model



@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model




@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model



@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(
        "vit_large_patch32_384", pretrained=pretrained, **model_kwargs
    )
    return model
