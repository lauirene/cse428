from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from model.pos_embed import get_2d_sincos_pos_embed,get_2d_sincos_pos_embed_rectangle

import numpy as np
from typing import Set

def unnormalize_image(samples):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imagenet_mean = torch.tensor(imagenet_mean,device=samples.device)
    imagenet_std = torch.tensor(imagenet_std,device=samples.device)
    new_samples = torch.einsum("bchw,c->bchw",samples,imagenet_std)
    new_samples = torch.clip((new_samples+ imagenet_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) * 255, 0, 255)
    return new_samples

class SmallSequenceCNN(nn.Module):
    def __init__(self, input_sequence_length: int, output_embed_dim: int, input_channels: int = 4):
        super().__init__()
        # Input: (Batch, Length, Channels=4) -> (Batch, Channels=4, Length) for Conv1d
        self.input_sequence_length = input_sequence_length
        self.output_embed_dim = output_embed_dim
        self.input_channels = input_channels # 4 for A,C,G,T

        # Simple CNN layers
        # Conv1d expects (Batch, In_Channels, Length)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=8, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(64, 128, kernel_size=8, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(128, 256, kernel_size=8, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            # Add more layers or adjust parameters as needed
        )

        # Calculate the output length after conv/pooling to determine the input for the linear layer
        # This is a rough calculation, you might need to run a dummy tensor through it
        # or use a more precise formula for padding='same' etc.
        # Let's assume output_length can be determined, or use AdaptiveMaxPool1d
        
        # Example to calculate output length:
        # L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1
        # For MaxPool1d: L_out = (L_in - kernel_size)/stride + 1
        
        # A safer approach for pooling to a fixed size is AdaptiveAvgPool1d
        self.global_pool = nn.AdaptiveAvgPool1d(1) # Pools across sequence length to get (Batch, Channels, 1)

        # Final linear layer to project to the desired embedding dimension
        # Input to linear: (Batch, Last_Conv_Channels)
        self.final_linear = nn.Linear(256, output_embed_dim)

    def forward(self, x):
        # Input x: (Batch, Length, Channels=4)
        # Transpose to (Batch, Channels, Length) for Conv1d
        x = x.transpose(1, 2) # (B, 4, L)
        
        x = self.conv_layers(x) # (B, 256, L_out)
        
        # Global pooling to get a fixed-size embedding for each sample
        x = self.global_pool(x).squeeze(-1) # (B, 256)
        
        # Project to the final embedding dimension
        x = self.final_linear(x) # (B, output_embed_dim)
        
        return x

class Finetune_Model_Head(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, vit_backbone,task=1,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,pos_embed_size=(1,250),
                 use_sequence=False):
        """
        task 0: fine-tuning setting
        task 1: reproducibility analysis
        task 2: loop calling
        task 3: resolution enhancement
        task 4: epigenomic assay prediction
        task 5: scHi-C enhancement
        task 6: embedding analysis
        task 7: reconstruction visualization (for pre-training only)
        """

        super().__init__()
        # --------------------------------------------------------------------------
        # HiCFoundation encoder 
        self.vit_backbone = vit_backbone
        self.embed_dim = vit_backbone.embed_dim
        self.task = task

        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer

        num_patches = self.vit_backbone.patch_embed.num_patches
        self.patch_embed = self.vit_backbone.patch_embed
        patch_size = self.vit_backbone.patch_size
        self.in_chans = self.vit_backbone.in_chans

        self.pos_embed_size = pos_embed_size
        
        # Conditionally instantiate SmallSequenceCNN
        self.sequence_cnn = None
        # self.sequence_cnn_embed_dim is now just self.embed_dim as it outputs directly to that dim
        
        if self.use_sequence:
            print("Sequence CNN component ENABLED: Instantiating SmallSequenceCNN.")
            # Assume sequence_length is available (e.g., from args or config if needed)
            # For now, hardcode sequence_length (e.g., 131072) for the SmallSequenceCNN init
            # You might need to make this configurable later if sequence_length varies.
            self.sequence_cnn = SmallSequenceCNN(
                input_sequence_length=131072, # From data_prep.py, ENFORMER_SEQUENCE_LENGTH
                output_embed_dim=self.embed_dim, # Output directly to ViT's embedding dim
                input_channels=4 # For one-hot A,C,G,T
            )
            # Initialize SmallSequenceCNN's weights (if not pre-trained)
            self._init_weights(self.sequence_cnn)
            print("SmallSequenceCNN loaded/initialized.")
            
            # For a small CNN, it's often trainable.
            # If you want it frozen for a direct comparison, you would freeze it here:
            # for param in self.sequence_cnn.parameters():
            #     param.requires_grad = False
            # print("SmallSequenceCNN parameters frozen.")
        else:
            print("Sequence CNN component DISABLED.")

        # HiCFoundation decoder 
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        
        
        if self.task==4:
            #for epigenomic assay prediction
            self.decoder_pos_embed_new = nn.Parameter(torch.zeros(1, num_patches , decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding, avoid loading from previous checkpoint
        else:
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        #different name to avoid loading weights error for this
        if self.task==2 or self.task==3 or self.task==5:
            #for loop calling and resolution enhancement and scHi-C enhancement
            self.decoder_map = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) # decoder to patch
        elif self.task==4:
            output_dim = patch_size
            num_track = 6
            #for epigenomic assay prediction
            self.map_blocks = []
            for k in range(num_track):
                map_block = nn.Linear(decoder_embed_dim, output_dim)
                self.map_blocks.append(map_block)
            self.map_blocks = nn.ModuleList(self.map_blocks)
        elif self.task==0:
            output_dim = patch_size
            self.decoder_map = nn.Linear(decoder_embed_dim, patch_size**2 * 1, bias=True) #map to 2d
            self.map_block = nn.Linear(decoder_embed_dim, output_dim) # map to 1d
        elif self.task==7:
            #for pre-train reconstruction visualization only
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3, bias=True)
        self.num_additional_token = 2 # 1 cls token and 1 count token
        # line below to account for added enformer token
        self.num_tokens_before_patches = self.num_additional_token + (1 if self.use_sequence else 0)
        self.initialize_weights()
    
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        no_wd = {'pos_embed', 'cls_token', 'dist_token'}
        if self.use_sequence and self.sequence_cnn.training:
            # Assuming SmallSequenceCNN does not have its own no_weight_decay method,
            # we don't explicitly add its parameters here unless needed.
            # By default, parameters of nn.Modules are included in weight decay unless specified.
            pass
        return no_wd
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        
        if self.task==4:
            decoder_pos_embed =get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed_new.shape[2], self.pos_embed_size, False)
            self.decoder_pos_embed_new.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            decoder_pos_embed =get_2d_sincos_pos_embed_rectangle(self.decoder_pos_embed.shape[2], self.pos_embed_size, False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self._init_weights(self.decoder_embed)
        self._init_weights(self.decoder_blocks)
        self._init_weights(self.decoder_norm)
        if self.task==4:
            for map_block in self.map_blocks:
                self._init_weights(map_block)
        # self._init_weights(self.decoder_map)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, h * p))
        return imgs

    def unpatchify_channel(self, x,in_chans):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1]**.5)
        # assert h * w == x.shape[1]
        h= self.pos_embed_size[0]
        w= self.pos_embed_size[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))
        return imgs
    #@torch.no_grad()
    def forward_backbone(self,img,total_count):
        img=self.vit_backbone.forward_features(img,total_count)
        return img
    def forward_decoder(self, img, sequence_data, total_count=None):
        if total_count is None:
            total_count = torch.ones(img.shape[0]).to(img.device)
            total_count = total_count*1000000000
        x = self.forward_backbone(img,total_count)
        # --- MODIFICATION START ---
        # Conditionally process sequence data for Sequence CNN token
        if self.use_sequence:
            # Check if sequence_data is valid when CNN is enabled
            if sequence_data is None:
                raise ValueError("Sequence data cannot be None when Sequence CNN is enabled (--use_enformer).")
            if self.sequence_cnn is None: # Defensive check
                 raise RuntimeError("Sequence CNN component is None when use_sequence_cnn is True. Check instantiation.")

            sequence_embedding = self.sequence_cnn(sequence_data) # (Batch, output_embed_dim)
            sequence_embedding_projected = sequence_embedding.unsqueeze(1) # (Batch, 1, output_embed_dim)

            # Concatenate Sequence CNN token after CLS and Count tokens
            x = torch.cat((x[:, :self.num_additional_token, :], sequence_embedding_projected, x[:, self.num_additional_token:, :]), dim=1)
        else:
            # If Sequence CNN is NOT used, the sequence remains as hic_embedding (CLS, Count, Patches).
            # No additional zero token is added here, so the sequence length changes.
            # This is the "true conditional instantiation" route.
            pass # No action needed, x remains as hic_embedding
        # --- MODIFICATION END ---
        
        if self.task==6:
            embedding_list = []
            embedding_list.append(x)
        # embed tokens
        x = self.decoder_embed(x)
        # add pos embed

        num_additional_token = self.num_additional_token # 1 cls token and 1 count token
        # append mask tokens to sequence
        # add pos embed
        if self.task==4:
            x[:,num_additional_token:] = x[:,num_additional_token:] + self.decoder_pos_embed_new
        else:
            x[:,num_additional_token:] = x[:,num_additional_token:] + self.decoder_pos_embed

        # we should not add count information to the decoder here, but we can use all-one embedding to distinguish the cls token and count_token
        x[:,1] = x[:,1]+1#all-one embedding for count token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
            if self.task==6:
                embedding_list.append(x)
        x = self.decoder_norm(x)
        if self.task==6:
            return embedding_list
        else:
            return x 
        
     

    
    def forward(self, img,total_count=None, sequence_data=None):
        """input hic image"""
        if self.task==0:
            #for fine-tuning
            decoder_output = self.forward_decoder(img, sequence_data=sequence_data,
                                                total_count=total_count)
            submatrix_embedding = decoder_output[:,0,:]
            pred_2d = self.decoder_map(decoder_output)
            pred_2d = pred_2d[:,self.num_tokens_before_patches:,:]
            pred_2d = self.unpatchify_channel(pred_2d,1)
            patch_embedding = decoder_output[:,self.num_tokens_before_patches:,:]
            num_patch_row = self.pos_embed_size[0]
            num_patch_col = self.pos_embed_size[1]
            pred_1d = patch_embedding.reshape(shape=(patch_embedding.shape[0], num_patch_row,num_patch_col,-1)) #average all columns
            pred_1d = torch.mean(pred_1d,dim=2) #N, H, C
            pred_1d = self.map_block(pred_1d) #N, H, D, where D is the output_dim
            pred_1d = pred_1d.reshape(pred_1d.shape[0],-1) #N, H*D
            return submatrix_embedding, pred_2d[:,0,:], pred_1d
        elif self.task==1:
            #for reproducibility analysis
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            return decoder_output[:,0,:]
        elif self.task==2 or self.task==3 or self.task==5:
            #for loop calling, resolution enhancement and scHi-C enhancement
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            decoder_output = self.decoder_map(decoder_output)
            # use patch-wise token
            decoder_output= decoder_output[:,self.num_tokens_before_patches:,:]
            pred_image = self.unpatchify_channel(decoder_output,1)
            return pred_image[:,0,:]
        
        elif self.task==7:
            #for pre-train reconstruction visualization only
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            decoder_output = self.decoder_pred(decoder_output)
            # use patch-wise token
            decoder_output= decoder_output[:,self.num_tokens_before_patches:,:]
            pred_image = self.unpatchify_channel(decoder_output,3)
            #remove the normatlization
            pred_image = unnormalize_image(pred_image)
            return pred_image

        
        elif self.task==4:
            #for epigenomic assay prediction
            decoder_output = self.forward_decoder(img,
                                                total_count=total_count)
            decoder_output = decoder_output[:,self.num_tokens_before_patches:,:]
            num_patch_row = self.pos_embed_size[0]
            num_patch_col = self.pos_embed_size[1]
            x = decoder_output.reshape(shape=(decoder_output.shape[0], num_patch_row,num_patch_col,-1)) #average all columns
            x = torch.mean(x,dim=2) #N, H, C
            #forward invidiual track's small block to map to different track's output
            output = []
            for map_block in self.map_blocks:
                y = x
                y = map_block(y) #N, H, D, where D is the output_dim
                y = y.reshape(y.shape[0],-1) #N, H*D
                output.append(y)
            output = torch.stack(output, dim=1) #change to N, num_track, C
            return output
        
        elif self.task==6:
            embedding_list = self.forward_decoder(img,
                                                total_count=total_count)
            #remove cls and additional token, which is not very useful in pre-training
            final_embedding = []
            for embedding in embedding_list:
                embedding = embedding[:,self.num_tokens_before_patches:,:]
                # shapre N, L, C
                # reshape to N, H,W,C
                num_patch_row = self.pos_embed_size[0]
                num_patch_col = self.pos_embed_size[1]
                embedding = embedding.reshape(shape=(embedding.shape[0], num_patch_row,num_patch_col,-1)) 
                final_embedding.append(embedding)
            return final_embedding

        else:
            print("Task ",self.task," is not implemented")
            print("Please specify the task using --task with 1,2,3,4,5,6")
            raise NotImplementedError(f"Task {self.task} is not implemented")