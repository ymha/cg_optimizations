# -----------------------------------------------------------------------------
# Copyright (C) 2024, Electronics and Telecommunications Research Institute (ETRI)
# All rights reserved.
#
# This code is a simple proof of concept based on the description in the paper:
# Gagrani et al., "Neural Topological Ordering for Computation Graphs", NeurIPS 2022.
#
# @Author: Youngmok Ha
#
# Date: September 27, 2024
# -----------------------------------------------------------------------------

import math
import torch
import torch.nn as nn

class FullyConnectedNetwork(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, dropout_rate):
        super(FullyConnectedNetwork, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, x):
        return self.ff(x)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_embed, dim_qkv, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        # Dimension of linear transformation (lt): (dim_embed, dim_qkv)
        self.lt_q = nn.Linear(dim_embed, dim_qkv)
        self.lt_k = nn.Linear(dim_embed, dim_qkv)
        self.lt_v = nn.Linear(dim_embed, dim_qkv)

        self.dim_embed = dim_embed
        self.dim_qkv = dim_qkv
        self.num_heads = num_heads
        self.dim_k = self.dim_qkv // num_heads
        assert (self.dim_k * self.num_heads == self.dim_qkv, "dim_qkv must be divisible by num_heads")
        self.scale = math.sqrt(self.dim_k)
        self.dropout = nn.Dropout(dropout_rate)

        self.lt_end = None if self.dim_embed == self.dim_qkv else nn.Linear(self.dim_qkv, self.dim_embed)

    def forward(self, x, x_mask=None, direction=1):
        """
        Method Description:

        Args:
            x (torch.Tensor): Input feature tensor with shape (batch_size, num_nodes, dim_embed).
            x_mask (torch.Tensor, optional): Mask matrix with shape (batch_size, num_cnode, num_cnode). Defaults to None.
            direction (int, optional): Forward attention flag. Defaults to 1.

        Dimensionality:
            x    : (batch_size, num_nodes, dim_embed)
            mask : (batch_size, num_nodes, num_nodes), optional
        """

        # Method logic here
        batch_size = x.size(0)
        num_nodes  = x.size(1)

        mask = x_mask

        # Perform linear transformation to get q, k, v from the input feature (x)
        # Shape transformation: (batch_size, num_nodes, dim_embed) -> (batch_size, num_nodes, dim_qkv)
        # This is done via w_q, w_k, w_v, each of shape (dim_embed, dim_qkv)
        q = self.lt_q(x)
        k = self.lt_k(x)
        v = self.lt_v(x)

        # Change the shape from (batch_size, num_nodes, dim_qkv) to (batch_size, num_heads, num_nodes, dim_k)
        if self.num_heads > 1:
            q = q.view(batch_size, -1, self.num_heads, self.dim_k)
            k = k.view(batch_size, -1, self.num_heads, self.dim_k)
            v = v.view(batch_size, -1, self.num_heads, self.dim_k)

            # double check of the dimension
            assert(q.size(1) == k.size(1) == v.size(1) == num_nodes)

            q = q.transpose(1,2)
            k = k.transpose(1,2)
            v = v.transpose(1,2)

            # shape change
            # from (batch_size, num_nodes, num_nodes) to (batch_size, num_heads, num_nodes, num_nodes)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1) if mask is not None else None

        # Compute attention scores
        # Score function: dot-product attention score
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Mask application (if provided)
        # Assumption: the mask matrix is designed for the forward direction.
        if mask is not None:
            if direction == 0:
                mask = mask.transpose(-2, -1)
            elif direction == 1:
                pass
            else:
                raise ValueError("Undefined mask direction. `direction` should be 1 or 0.")
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Get attention weights via softmax
        # Dimension: (batch_size, num_heads, num_nodes, num_nodes)
        # Softmax and dropout functions do not change the shape of the tensor
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Get updated feature z
        # Dimension of z (single head): (batch_size, num_nodes, dim_qkv)
        # Dimension of z (multi-head) : (batch_size, num_heads, num_nodes, dim_k)
        # = (batch_size, num_heads, num_nodes, num_nodes) * (batch_size, num_heads, num_nodes, dim_k)
        z = torch.matmul(attention_weights, v)

        if self.num_heads > 1:
            # Concatenate results from multiple heads
            # Dimension of z (after) : (batch_size, num_nodes, dim_qkv)
            # Dimension of z (before): (batch_size, num_heads, num_nodes, dim_k)
            z = z.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_qkv)

        assert (z.size(1) == num_nodes)

        y = self.lt_end(z) if self.lt_end is not None else z

        return y

class InputEmbeddingLayer(nn.Module):
    def __init__(self, dim_input, dim_embed, num_graphs, dropout_rate):
        super(InputEmbeddingLayer, self).__init__()
        self.dim_input = dim_input
        self.lts = nn.ModuleList([nn.Linear(dim_input, dim_embed) for _ in range(num_graphs)])
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        assert(x.size(-1) // len(self.lts) == self.dim_input)
        x_split = torch.split(x, self.dim_input, dim=-1)
        z_list = [self.dropout(lt_i(x_i)) for lt_i, x_i in zip(self.lts, x_split)]

        # Dimension of y: (batch_size, num_nodes, dim_embed * 7)
        y = torch.cat(z_list, dim=-1)
        return y

class SubLayerBlock(nn.Module):
    def __init__(self, dim_embed, dim_qkv, num_heads, dropout_rate, norm_epsilon):
        super(SubLayerBlock, self).__init__()
        self.mha = MultiHeadAttention(dim_embed, dim_qkv, num_heads, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(dim_embed, eps=norm_epsilon)

    def forward(self, x, x_mask, direction=1):
        z = self.mha(x, x_mask, direction)
        z = self.dropout(z)
        y = self.norm(z + x)  # add & norm
        return y

class Layer(nn.Module):
    def __init__(self, num_blocks, dim_embed, dim_qkv, num_heads, dim_hidden, dropout_rate, norm_epsilon):
        super(Layer, self).__init__()
        # 1. Transitive reduction (TR) of DAG
        # 2. Backward version of 1
        # 3. The directed graph obtained by removing the TR edges from the DAG
        # 4. Backward version of 3
        # 5. The directed graph obtained by removing the edges of the DAG from its TC
        # 6. Backward version of 5
        # 7. The undirected graph obtained by joining all incomparable node pairs.

        self.dim_embed = dim_embed
        self.num_blocks = num_blocks

        # Initialize normalization layers for each block
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim_embed, eps=norm_epsilon)
            for _ in range(num_blocks)])

        # Initialize sublayer blocks for each block
        self.blocks = nn.ModuleList([
            SubLayerBlock(dim_embed, dim_qkv, num_heads, dropout_rate, norm_epsilon)
            for _ in range(num_blocks)
        ])

        # Fully connected network (FCN) for final output
        self.fcn = FullyConnectedNetwork(dim_embed * num_blocks, dim_hidden, dim_embed * num_blocks, dropout_rate)

        # Create alternating directions (1 for forward, 0 for backward)
        self.direction = torch.remainder(torch.arange(1, self.num_blocks + 1),2)

    def forward(self, x, x_mask):
        # Dimension of x: (batch_size, num_nodes, dim_embed * num_blocks)
        # Dimension of x_mask: (batch_size, num_nodes, num_nodes * num_blocks)
        dim_embed = x.size(-1) // len(self.blocks)
        num_nodes = x_mask.size(-2)

        # Dimension of x_split: (batch_size, num_nodes, dim_embed)
        # Dimension of x_mask_split: (batch_size, num_nodes, num_nodes)
        x_split = torch.split(x, dim_embed, dim=-1)
        x_mask_split = torch.split(x_mask, num_nodes, dim=-1)

        # Process each block with its corresponding split and mask
        z_list = [
            block(self.norms[i](x_i), x_mask_i, direction=self.direction[i])
            for i, (block, x_i, x_mask_i) in enumerate(zip(self.blocks, x_split, x_mask_split))
        ]

        # Concatenate the outputs from all blocks along the last dimension.
        # Dimension of z: (batch_size, num_nodes, dim_embed * num_blocks)
        z = torch.cat(z_list, dim=-1)

        # Apply the fully connected network (FCN) to the combined output
        y = self.fcn(x+z)

        return y


class Topoformer(nn.Module):
    # the structure and parameters are set based on the paper.
    # Notice: we have to check the number of hidden neurons used for equation (5) in the paper
    def __init__(self,
                 dim_input=20,
                 dim_embed=256,
                 num_layers=4,
                 num_blocks=7,
                 dim_qkv=640,
                 num_heads=10,
                 dim_hidden=1792,
                 dropout_rate=0.0,
                 norm_epsilon=1e-8):
        super(Topoformer, self).__init__()
        self.input_embedding = InputEmbeddingLayer(dim_input, dim_embed, num_blocks, dropout_rate)
        self.layers = nn.ModuleList([
            Layer(num_blocks, dim_embed, dim_qkv, num_heads, dim_hidden, dropout_rate, norm_epsilon)
            for _ in range(num_layers)
        ])
        self.lt = nn.Linear(dim_embed*num_blocks, dim_input)

    def forward(self, x, x_mask):
        z = self.input_embedding(x)
        for layer in self.layers:
            z = layer(z, x_mask)
        y = self.lt(z)
        return y
