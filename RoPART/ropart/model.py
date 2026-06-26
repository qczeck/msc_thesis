"""Focused, self-contained ViT for the RoPART pretext task.

The building blocks (``PatchEmbed``, ``Mlp``, ``Attention`` with optional key
masking, ``Block``, ``CrossAttention``) are copied from
``PART/timm/models/vision_transformer.py`` so RoPART has no ``timm`` dependency
(``DropPath`` is reimplemented; ``trunc_normal_`` comes from ``torch.nn.init``).

``RoPARTViT`` keeps only what we need:

* **pretrain** forward — encoder **without position embeddings** + cls token,
  then a ``CrossAttention`` relative head over a random subset of ordered patch
  pairs → ``(outputs [b, C, num_pairs], indices [2, num_pairs])``;
* **classification** forward — same encoder *with* position embeddings → cls
  token → linear ``clf`` (for the linear-probe / finetune).

Dropped vs upstream: v1/v2 grid sampling, the mlp / pairwise_mlp heads, column
embeddings, hybrid backbones.
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


# --------------------------------------------------------------------------- #
# Building blocks (copied from upstream timm vision_transformer)
# --------------------------------------------------------------------------- #


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    """Self-attention with optional restriction of keys/values to ``key_ind``.

    With ``key_ind=None`` (the default, and the case for ``mask_prob=0``) every
    token attends to every token — a standard ViT block.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, key_ind=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if key_ind is not None:
            xkv = x[torch.arange(B).unsqueeze(1), key_ind]
            Nkv = key_ind.size(1)
        else:
            xkv = x
            Nkv = N
        k, v = self.kv(xkv).reshape(B, Nkv, self.num_heads, 2 * C // self.num_heads).permute(0, 2, 1, 3).chunk(2, dim=-1)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0,
                 attn_drop=0.0, drop_path_=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_) if drop_path_ > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, key_ind=None):
        x = x + self.drop_path(self.attn(self.norm1(x), key_ind))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to patch embedding (conv with stride = patch size)."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"input {(H, W)} != model {self.img_size}"
        return self.proj(x).flatten(2).transpose(1, 2)


class CrossAttention(nn.Module):
    """Relative encoder: cross-attend pair queries over the patch tokens.

    Copied from upstream. Query is either a fixed 2-D positional encoding of the
    pair ``(i, j)`` (``positional``) or the two patch features concatenated and
    projected (``patch_cat``). Output is ``num_channels`` per pair.
    """

    def __init__(self, embed_dim, num_channels, num_patches, num_heads=1, query_type="positional"):
        super().__init__()

        def positionalencoding2d(d_model, height, width):
            if d_model % 4 != 0:
                raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got {d_model})")
            pe = torch.zeros(d_model, height, width)
            d_model = int(d_model / 2)
            div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
            pos_w = torch.arange(0.0, width).unsqueeze(1)
            pos_h = torch.arange(0.0, height).unsqueeze(1)
            pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
            pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
            return pe.permute(1, 2, 0)

        def positionalencoding1d(d_model, length):
            if d_model % 2 != 0:
                raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got {d_model})")
            pe = torch.zeros(length, d_model)
            position = torch.arange(0, length).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            return pe

        self.register_buffer("positional_embeddings2d", positionalencoding2d(embed_dim, num_patches, num_patches))
        self.register_buffer("positional_embeddings1d", positionalencoding1d(embed_dim, num_patches))
        self.num_channels = num_channels
        self.query_type = query_type
        self.input_project = nn.Linear(embed_dim * 2, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.output_project = nn.Linear(embed_dim, num_channels)

    def forward(self, x, query):
        """x: [bs, num_patches, embed]; query: [bs, num_pairs, 2] -> [bs, num_pairs, num_channels]."""
        bs, num_patches, _ = x.shape
        query = query.to(x.device)
        if self.query_type == "positional":
            transformed_queries = self.positional_embeddings2d[query[:, :, 0], query[:, :, 1], :]
            key_pos = self.positional_embeddings1d[range(num_patches), :].repeat(bs, 1, 1)
            x = x + key_pos
        elif self.query_type == "patch_cat":
            num_pairs = query.shape[1]
            batch_indice = torch.arange(bs)[..., None].repeat(1, num_pairs)
            q0 = x[batch_indice, query[:, :, 0], :]
            q1 = x[batch_indice, query[:, :, 1], :]
            transformed_queries = self.input_project(torch.cat([q0, q1], dim=2))
        attn_output, _ = self.multihead_attn(query=transformed_queries, key=x, value=x)
        return self.output_project(attn_output)


# --------------------------------------------------------------------------- #
# RoPART ViT
# --------------------------------------------------------------------------- #


class RoPARTViT(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 100,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_channels: int = 2,
        num_pairs: int = 64,
        mask_prob: float = 0.0,
        use_pe: bool = True,
        cross_attention_num_heads: int = 1,
        cross_attention_query_type: str = "positional",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_pairs = num_pairs
        self.mask_prob = mask_prob
        self.use_pe = use_pe
        self.embed_dim = embed_dim
        self.depth = depth

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, drop=drop_rate,
                  drop_path_=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # relative head (pretrain) and classification head (probe/finetune)
        self.head = CrossAttention(embed_dim, num_channels, self.num_patches,
                                   cross_attention_num_heads, cross_attention_query_type)
        self.clf = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _encode(self, x, *, add_pos: bool, mask: bool):
        """Run the encoder. Returns ``(B, num_patches + 1, embed)`` (cls last)."""
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed)
        key_ind = None
        if mask and self.mask_prob > 0:
            n_keys = int(self.num_patches * (1 - self.mask_prob))
            _, ind = torch.randn(B, self.num_patches, device=x.device).sort(dim=-1)
            cls_ind = torch.full((B, 1), self.num_patches, device=x.device, dtype=torch.long)
            key_ind = torch.cat([ind[:, :n_keys], cls_ind], dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((x, cls), dim=1)  # cls at the end
        if add_pos:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x, key_ind)
        return self.norm(x)

    def forward_pretrain(self, x):
        """Relative pretext forward: -> (outputs [b, C, num_pairs], indices [2, num_pairs])."""
        feats = self._encode(x, add_pos=False, mask=True)[:, :-1, :]  # drop cls
        b, n, _ = feats.shape
        idx = torch.randperm(n * n, device=x.device)[: self.num_pairs][:, None]
        x_idx, y_idx = idx % n, idx // n
        pairs = torch.cat((y_idx, x_idx), dim=1).repeat(b, 1, 1)  # (b, num_pairs, 2) = (ref i, tgt j)
        out = self.head(feats, pairs).permute(0, 2, 1)  # (b, C, num_pairs)
        indices = pairs.permute(0, 2, 1)[0]  # (2, num_pairs)
        return out, indices

    def forward_classify(self, x):
        """Classification forward (probe / finetune): -> logits [b, num_classes]."""
        cls = self._encode(x, add_pos=self.use_pe, mask=False)[:, -1]
        return self.clf(cls)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
#
# Maps a model name to its *architecture* dims (image/patch geometry + ViT width
# and depth). The pretext-specific kwargs (``num_channels``, ``num_pairs``,
# ``mask_prob``, ``num_classes``, …) are supplied by the caller, not here. Adding a
# config is a one-line entry; ``model_config`` is the single place that turns a
# name into geometry, so callers never re-parse the string (the old
# ``train.parse_model_name`` mis-parsed two-digit patch sizes, e.g. ``patch16``).

MODEL_CONFIGS: dict[str, dict] = {
    # name: (img_size, patch_size, embed_dim, depth, num_heads)
    "deit_small_patch4_32": dict(img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=6),
    "deit_small_patch8_32": dict(img_size=32, patch_size=8, embed_dim=384, depth=12, num_heads=6),
    "deit_small_patch16_224": dict(img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6),
    "deit_base_patch16_224": dict(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12),
}


def model_config(name: str) -> dict:
    """Return the architecture dims for a registered model name.

    Args:
        name: a key of :data:`MODEL_CONFIGS`, e.g. ``"deit_base_patch16_224"``.

    Returns:
        A fresh dict with ``img_size, patch_size, embed_dim, depth, num_heads``.
    """
    if name not in MODEL_CONFIGS:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODEL_CONFIGS)}")
    return dict(MODEL_CONFIGS[name])


def build_model(name: str, **kwargs) -> RoPARTViT:
    """Construct a :class:`RoPARTViT` from a registered name plus pretext kwargs.

    ``kwargs`` (e.g. ``num_classes``, ``num_channels``, ``num_pairs``, ``mask_prob``,
    ``drop_path_rate``, ``cross_attention_query_type``) override / extend the
    architecture dims from :func:`model_config`.
    """
    return RoPARTViT(**model_config(name), **kwargs)


def deit_small_patch4_32(**kwargs) -> RoPARTViT:
    """ViT-S/4 for 32x32 — the CIFAR-100 workhorse (matches upstream config)."""
    return build_model("deit_small_patch4_32", **kwargs)


def deit_small_patch16_224(**kwargs) -> RoPARTViT:
    """ViT-S/16 for 224x224 — light ImageNet option."""
    return build_model("deit_small_patch16_224", **kwargs)


def deit_base_patch16_224(**kwargs) -> RoPARTViT:
    """ViT-B/16 for 224x224 — the ImageNet workhorse (matches the source paper)."""
    return build_model("deit_base_patch16_224", **kwargs)
