import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vivit_header import Attention, PreNorm, FeedForward
import numpy as np

class MLP_CLS(nn.Module):
    def __init__(self, num_classes=2, e2e=False):
        super().__init__()
        self.e2e = e2e
        self.mlp = nn.Sequential(
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward_once(self, x):
        return self.mlp(x)

    def forward(self, x):
        if not self.e2e:
            return self.forward_once(x)
        else:
            return self.forward_once(x[0]), self.forward_once(x[1])

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(self, clip_frames, image_size, patch_size, num_classes, T_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, reverse_axis=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.patch_size = patch_size
        self.image_size = image_size
        self.reverse_axis = reverse_axis


        patch_dim = in_channels * patch_size ** 2 * T_frames
        self.nt = clip_frames // T_frames
        self.reverse_axis = reverse_axis

        if not self.reverse_axis:
            num_patches = (image_size // patch_size) ** 2
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b (t pt) c (h p1) (w p2) -> b t (h w) (pt p1 p2 c)', pt = T_frames, p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
            self.pos_embedding = nn.Parameter(torch.randn(1, self.nt, num_patches + 1, dim))

        else:
            num_patches = (image_size // patch_size) * (clip_frames // T_frames)
            self.nt = image_size // patch_size
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b (w p2) c (h p1) (t pt) -> b w (h t) (p2 p1 pt c)', p2 = patch_size, p1 = patch_size, pt = T_frames),
                nn.Linear(patch_dim, dim),
            )
            self.pos_embedding = nn.Parameter(torch.randn(1, self.nt, num_patches + 1, dim))

        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

#        self.mlp_head = nn.Sequential(
#            nn.LayerNorm(dim),
#            nn.Linear(dim, num_classes),
#        )

    def forward_once(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        if not self.reverse_axis:
            cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
            cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        else:
            w = self.image_size // self.patch_size
            cls_space_tokens = repeat(self.space_token, '() n d -> b w n d', b = b, w=w)
            cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)


        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')

        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return x

    def forward(self, x1, x2=None, training=True):
        if training:
            fea1 = self.forward_once(x1)
            fea2 = self.forward_once(x2)
            distance = F.pairwise_distance(fea1, fea2)
            return distance, fea1, fea2

        else:
            feature = self.forward_once(x1)
            return feature
