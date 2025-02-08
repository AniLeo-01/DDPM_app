import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from .utils import *

class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
  """
  Implements Weight Standardized Conv2D
  """

  def forward(self, x):
    eps = 1e-5 if x.dtype == torch.float32 else 1e-3

    # Compute mean and variance along output channels
    mean = self.weight.mean(dim=[1, 2, 3], keepdim=True)
    var = self.weight.var(dim=[1, 2, 3], unbiased=False, keepdim=True)

    # Normalize weights
    normalized_weight = (self.weight - mean) / torch.sqrt(var + eps)

    return F.conv2d(
      x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups
    )
  
class Block(nn.Module):
  def __init__(self, dim, dim_out, groups=8):
    super().__init__()

    # Ensure groups is a divisor of dim_out
    while dim_out % groups != 0 and groups > 1:
      groups -= 1  # Reduce groups dynamically

    self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
    self.norm = nn.GroupNorm(groups, dim_out)
    self.act = nn.SiLU()

  def forward(self, x, scale_shift=None):
    x = self.proj(x)
    x = self.norm(x)

    if scale_shift is not None:
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    x = self.act(x)
    return x

    
class ResnetBlock(nn.Module):
  def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
    super().__init__()

    # Reduce groups dynamically if necessary
    while dim_out % groups != 0 and groups > 1:
      groups -= 1

    self.mlp = (
      nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
      if time_emb_dim is not None
      else None
    )

    self.block1 = Block(dim, dim_out, groups=groups)
    self.block2 = Block(dim_out, dim_out, groups=groups)
    self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

  def forward(self, x, time_emb=None):
    scale_shift = None
    if self.mlp is not None and time_emb is not None:
      time_emb = self.mlp(time_emb)
      time_emb = time_emb[:, :, None, None]  # Expand dimensions without einops
      scale_shift = time_emb.chunk(2, dim=1)

    h = self.block1(x, scale_shift=scale_shift)
    h = self.block2(h)
    return h + self.res_conv(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from src.utils import *

class LinearAttention(nn.Module):
  def __init__(self, channels, groups=8):
    super().__init__()
    while channels % groups != 0 and groups > 1:
      groups -= 1  # Ensure channels is divisible by groups
    self.norm = nn.GroupNorm(groups, channels)
    self.qkv = nn.Linear(channels, channels * 3)
    self.proj = nn.Linear(channels, channels)

  def forward(self, x):
    B, C, H, W = x.shape
    x = self.norm(x)

    # Flatten spatial dimensions
    x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)

    # Compute Q, K, V
    qkv = self.qkv(x)  # Shape: (B, H*W, 3C)
    q, k, v = qkv.chunk(3, dim=-1)

    # Apply Softmax along K for linear attention
    k = k.softmax(dim=-1)
    q = q.softmax(dim=-2)

    # Compute linearized attention
    context = k.transpose(-2, -1) @ v  # Shape: (B, C, C)
    out = q @ context  # Apply context on queries
    out = self.proj(out)  # Shape: (B, H*W, C)

    # Reshape back
    return out.permute(0, 2, 1).view(B, C, H, W) + x.permute(0, 2, 1).view(B, C, H, W)

class MultiheadSelfAttention(nn.Module):
  def __init__(self, channels, num_heads=4):
    super().__init__()
    assert channels % num_heads == 0, "Channels must be divisible by num_heads"
    while channels % 32 != 0 and 32 > 1:
      groups = 32
      groups -= 1  # Ensure channels is divisible by groups
    self.num_heads = num_heads
    self.head_dim = channels // num_heads
    self.norm = nn.GroupNorm(groups, channels)
    
    # Projection layers for Q, K, V
    self.qkv = nn.Linear(channels, channels * 3)
    self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
    self.proj = nn.Linear(channels, channels)

  def forward(self, x):
    B, C, H, W = x.shape
    x = self.norm(x)

    # Flatten spatial dimensions
    x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)

    # Compute Q, K, V
    qkv = self.qkv(x)  # Shape: (B, H*W, 3C)
    q, k, v = qkv.chunk(3, dim=-1)

    # Apply multi-head attention
    attn_out, _ = self.attn(q, k, v)
    attn_out = self.proj(attn_out)  # Shape: (B, H*W, C)

    # Reshape back to original spatial dimensions
    return attn_out.permute(0, 2, 1).view(B, C, H, W) + x.permute(0, 2, 1).view(B, C, H, W)
  
class LinearAttention(nn.Module):
  def __init__(self, channels, groups=8):
    super().__init__()
    while channels % groups != 0 and groups > 1:
      groups -= 1  # Ensure channels is divisible by groups
    self.norm = nn.GroupNorm(groups, channels)
    self.qkv = nn.Linear(channels, channels * 3)
    self.proj = nn.Linear(channels, channels)

  def forward(self, x):
    B, C, H, W = x.shape
    x = self.norm(x)

    # Flatten spatial dimensions
    x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)

    # Compute Q, K, V
    qkv = self.qkv(x)  # Shape: (B, H*W, 3C)
    q, k, v = qkv.chunk(3, dim=-1)

    # Apply Softmax along K for linear attention
    k = k.softmax(dim=-1)
    q = q.softmax(dim=-2)

    # Compute linearized attention
    context = k.transpose(-2, -1) @ v  # Shape: (B, C, C)
    out = q @ context  # Apply context on queries
    out = self.proj(out)  # Shape: (B, H*W, C)

    # Reshape back
    return out.permute(0, 2, 1).view(B, C, H, W) + x.permute(0, 2, 1).view(B, C, H, W)
  
class UNet(nn.Module):
  def __init__(
    self,
    dim,
    init_dim=None,
    out_dim=None,
    dim_mults=(1, 2, 4, 8),
    channels=3,
    with_time_emb=True,
    resnet_block_groups=8,
  ):
    super().__init__()

    # determine dimensions
    self.channels = channels

    init_dim = default(init_dim, dim // 3 * 2)
    self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

    dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
    in_out = list(zip(dims[:-1], dims[1:]))
    
    block_klass = partial(ResnetBlock, groups=resnet_block_groups)

    # time embeddings
    if with_time_emb:
      time_dim = dim * 4
      self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(dim),
        nn.Linear(dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim),
      )
    else:
      time_dim = None
      self.time_mlp = None

    # layers
    self.downs = nn.ModuleList([])
    self.ups = nn.ModuleList([])
    num_resolutions = len(in_out)

    for ind, (dim_in, dim_out) in enumerate(in_out):
      is_last = ind >= (num_resolutions - 1)

      self.downs.append(
        nn.ModuleList(
          [
            block_klass(dim_in, dim_out, time_emb_dim=time_dim),
            block_klass(dim_out, dim_out, time_emb_dim=time_dim),
            Residual(PreNorm(dim_out, LinearAttention(dim_out))),
            Downsample(dim_out) if not is_last else nn.Identity(),
          ]
        )
      )

    mid_dim = dims[-1]
    self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
    self.mid_attn = Residual(PreNorm(mid_dim, MultiheadSelfAttention(mid_dim)))
    self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

    for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
      is_last = ind >= (num_resolutions - 1)

      self.ups.append(
        nn.ModuleList(
          [
            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
            block_klass(dim_in, dim_in, time_emb_dim=time_dim),
            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
            Upsample(dim_in) if not is_last else nn.Identity(),
          ]
        )
      )

    out_dim = default(out_dim, channels)
    self.final_conv = nn.Sequential(
      block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
    )

  def forward(self, x, time):
    x = self.init_conv(x)

    t = self.time_mlp(time) if exists(self.time_mlp) else None

    h = []

    # downsample
    for block1, block2, attn, downsample in self.downs:
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      h.append(x)
      x = downsample(x)

    # bottleneck
    x = self.mid_block1(x, t)
    x = self.mid_attn(x)
    x = self.mid_block2(x, t)

    # upsample
    for block1, block2, attn, upsample in self.ups:
      x = torch.cat((x, h.pop()), dim=1)
      x = block1(x, t)
      x = block2(x, t)
      x = attn(x)
      x = upsample(x)

    return self.final_conv(x)