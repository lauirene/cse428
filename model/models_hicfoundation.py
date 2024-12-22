
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# AdPE: https://github.com/maple-research-lab/AdPE
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block,PatchEmbed
import numpy as np

from model.pos_embed import get_2d_sincos_pos_embed_rectangle,convert_count_to_pos_embed_cuda



class Models_HiCFoundation(nn.Module):
    """ 
    HiCFoundation:
    Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, input_row_size=224, input_col_size=224,
                 patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        
        #encoder specification
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        #configure positional embedding
        self.img_size = (input_row_size, input_col_size)
        self.pos_embed_size = (input_row_size // patch_size, input_col_size // patch_size)
        self.patch_embed  = PatchEmbed(self.img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        #configure encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # decoder specification
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        self.decoder_count = nn.Linear(decoder_embed_dim, 1, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed= get_2d_sincos_pos_embed_rectangle(self.pos_embed.shape[2], self.pos_embed_size, True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed.shape[2], (self.pos_embed_size[0], self.pos_embed_size[1]), False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x,  mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        mask_ratio: float, masking ratio
        symmetrical mask in 2d space
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        pos_row_size, pos_col_size = self.pos_embed_size
        
        noise = torch.rand(N, pos_row_size,pos_col_size, device=x.device)
        noise = noise.view(N,L)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # self.len_keep = len_keep
        # self.mask = mask
        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs, total_count=None, mask_ratio=0.75):
        """
        imgs: [N, 3, H, W]

        total_count: [N, 1] total count of Hi-C, serve as input to predict the submatrix count
        """
        B, C, H, W = imgs.shape
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        if total_count is None:
            #placeholder if total_count is not provided
            total_count = torch.ones(imgs.shape[0]).to(imgs.device)
            total_count = total_count*1000000000
        # gen count embedding
        total_count = torch.log10(total_count)
        count_embed = convert_count_to_pos_embed_cuda(total_count, self.embed_dim)
        count_embed = count_embed.unsqueeze(1)# (N, 1, D)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, count_embed, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward(self, imgs, imgs_mask, total_count=None, mask_ratio=0.75):
        """
        imgs: [N, 3, H, W]
        imgs_mask: [N, 1, H, W] indicate those 0 regions and mask them in target
        total_count: [N, 1] total count of Hi-C, serve as input to predict the submatrix count
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, total_count,mask_ratio)

