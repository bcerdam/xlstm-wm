import torch
import torch.nn as nn
from typing import List
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig)

class XLSTM_DM(nn.Module):
    def __init__(self, 
                 sequence_length:int, 
                 num_blocks:int, 
                 embedding_dim:int, 
                 slstm_at:List, 
                 dropout:float, 
                 add_post_blocks_norm:bool, 
                 conv1d_kernel_size:int, 
                 qkv_proj_blocksize:int, 
                 num_heads:int) -> None:
        super().__init__()

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=conv1d_kernel_size, 
                    qkv_proj_blocksize=qkv_proj_blocksize, 
                    num_heads=num_heads)),
            context_length=sequence_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
            dropout=dropout,
            add_post_blocks_norm=add_post_blocks_norm)
        self.xlstm_stack = xLSTMBlockStack(cfg)

        # To do: Add mlp heads


    def forward():
        pass