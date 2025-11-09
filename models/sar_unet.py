from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel


@dataclass
class SarUNetConfig:
    """用来在 checkpoint 中保存模型关键超参数。"""

    image_size: int
    in_channels: int
    base_channels: int
    block_out_channels: Tuple[int, ...]
    down_block_types: Tuple[str, ...]
    up_block_types: Tuple[str, ...]
    layers_per_block: int
    cond_dim: int
    embed_dim_each: int
    num_classes: int
    num_angles: int
    num_jam_a: int
    num_jam_p: int
    attention_head_dim: int

    def to_dict(self):
        return {
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "block_out_channels": list(self.block_out_channels),
            "down_block_types": list(self.down_block_types),
            "up_block_types": list(self.up_block_types),
            "layers_per_block": self.layers_per_block,
            "cond_dim": self.cond_dim,
            "embed_dim_each": self.embed_dim_each,
            "num_classes": self.num_classes,
            "num_angles": self.num_angles,
            "num_jam_a": self.num_jam_a,
            "num_jam_p": self.num_jam_p,
            "attention_head_dim": self.attention_head_dim,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            image_size=int(data["image_size"]),
            in_channels=int(data["in_channels"]),
            base_channels=int(data["base_channels"]),
            block_out_channels=tuple(data["block_out_channels"]),
            down_block_types=tuple(data["down_block_types"]),
            up_block_types=tuple(data["up_block_types"]),
            layers_per_block=int(data["layers_per_block"]),
            cond_dim=int(data["cond_dim"]),
            embed_dim_each=int(data["embed_dim_each"]),
            num_classes=int(data["num_classes"]),
            num_angles=int(data["num_angles"]),
            num_jam_a=int(data["num_jam_a"]),
            num_jam_p=int(data["num_jam_p"]),
            attention_head_dim=int(data["attention_head_dim"]),
        )


class SarConditionalUNet(nn.Module):
    """带多条件嵌入的 UNet 模型。

    与官方示例不同，这里我们使用 ``UNet2DConditionModel`` 并通过 cross-attention 注入条件信息。
    四类条件（类别、角度、有源/无源干扰）分别嵌入，再拼接后投影成 cross-attention 所需的 hidden states。
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        in_channels: int = 1,
        base_channels: int = 128,
        num_res_blocks: int = 2,
        cond_dim: int = 256,
        embed_dim_each: int = 128,
        num_classes: int,
        num_angles: int,
        num_jam_a: int,
        num_jam_p: int,
        block_out_channels: Optional[Sequence[int]] = None,
        down_block_types: Optional[Sequence[str]] = None,
        up_block_types: Optional[Sequence[str]] = None,
        attention_head_dim: int = 8,
    ) -> None:
        super().__init__()

        if block_out_channels is None:
            block_out_channels = (
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 4,
            )
        if down_block_types is None:
            down_block_types = (
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            )
        if up_block_types is None:
            up_block_types = (
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            )

        self.config = SarUNetConfig(
            image_size=image_size,
            in_channels=in_channels,
            base_channels=base_channels,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=num_res_blocks,
            cond_dim=cond_dim,
            embed_dim_each=embed_dim_each,
            num_classes=num_classes,
            num_angles=num_angles,
            num_jam_a=num_jam_a,
            num_jam_p=num_jam_p,
            attention_head_dim=attention_head_dim,
        )

        # 四个条件分别 embedding
        self.class_embedding = nn.Embedding(num_classes, embed_dim_each)
        self.angle_embedding = nn.Embedding(num_angles, embed_dim_each)
        self.jam_a_embedding = nn.Embedding(num_jam_a, embed_dim_each)
        self.jam_p_embedding = nn.Embedding(num_jam_p, embed_dim_each)

        self.condition_proj = nn.Linear(embed_dim_each * 4, cond_dim)

        self.unet = UNet2DConditionModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=in_channels,
            block_out_channels=tuple(block_out_channels),
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            layers_per_block=num_res_blocks,
            cross_attention_dim=cond_dim,
            attention_head_dim=attention_head_dim,
        )

        self._init_embeddings()

    def _init_embeddings(self) -> None:
        # 使用较小的初始化，保证训练稳定
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        nn.init.normal_(self.angle_embedding.weight, std=0.02)
        nn.init.normal_(self.jam_a_embedding.weight, std=0.02)
        nn.init.normal_(self.jam_p_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.condition_proj.weight)
        nn.init.zeros_(self.condition_proj.bias)

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        class_id: torch.Tensor,
        angle_id: torch.Tensor,
        jam_a_id: torch.Tensor,
        jam_p_id: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回噪声预测。"""

        class_embed = self.class_embedding(class_id)
        angle_embed = self.angle_embedding(angle_id)
        jam_a_embed = self.jam_a_embedding(jam_a_id)
        jam_p_embed = self.jam_p_embedding(jam_p_id)

        # 拼接四类 embedding，并通过线性层投影
        cond_embed = torch.cat([class_embed, angle_embed, jam_a_embed, jam_p_embed], dim=-1)
        cond_embed = self.condition_proj(cond_embed)
        encoder_hidden_states = cond_embed.unsqueeze(1)

        outputs = self.unet(sample, timesteps, encoder_hidden_states=encoder_hidden_states)
        return outputs.sample

    @classmethod
    def from_config(cls, config: SarUNetConfig) -> "SarConditionalUNet":
        return cls(
            image_size=config.image_size,
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            num_res_blocks=config.layers_per_block,
            cond_dim=config.cond_dim,
            embed_dim_each=config.embed_dim_each,
            num_classes=config.num_classes,
            num_angles=config.num_angles,
            num_jam_a=config.num_jam_a,
            num_jam_p=config.num_jam_p,
            block_out_channels=config.block_out_channels,
            down_block_types=config.down_block_types,
            up_block_types=config.up_block_types,
            attention_head_dim=config.attention_head_dim,
        )
