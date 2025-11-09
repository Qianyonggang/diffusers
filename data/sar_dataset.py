import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class SARDatasetMetadata:
    """用于记录 SAR 数据集的条件维度信息，方便保存到 checkpoint 中。"""

    num_classes: int
    num_angles: int
    num_jam_a: int
    num_jam_p: int
    image_size: int
    class_to_id: Dict[str, int]
    angle_to_id: Dict[str, int]
    jam_a_to_id: Dict[str, int]
    jam_p_to_id: Dict[str, int]
    angle_bin_size: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_classes": self.num_classes,
            "num_angles": self.num_angles,
            "num_jam_a": self.num_jam_a,
            "num_jam_p": self.num_jam_p,
            "image_size": self.image_size,
            "class_to_id": self.class_to_id,
            "angle_to_id": self.angle_to_id,
            "jam_a_to_id": self.jam_a_to_id,
            "jam_p_to_id": self.jam_p_to_id,
            "angle_bin_size": self.angle_bin_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SARDatasetMetadata":
        return cls(
            num_classes=int(data["num_classes"]),
            num_angles=int(data["num_angles"]),
            num_jam_a=int(data["num_jam_a"]),
            num_jam_p=int(data["num_jam_p"]),
            image_size=int(data["image_size"]),
            class_to_id=dict(data["class_to_id"]),
            angle_to_id=dict(data["angle_to_id"]),
            jam_a_to_id=dict(data["jam_a_to_id"]),
            jam_p_to_id=dict(data["jam_p_to_id"]),
            angle_bin_size=data.get("angle_bin_size"),
        )


class SARDataset(Dataset):
    """SAR 图像数据集。

    该数据集会在初始化阶段递归遍历根目录下所有 ``.png`` 文件，解析文件名中的条件信息。
    文件名格式为 ``class_angle_jamActive_jamPassive.png``，其中类别可能包含 ``_``，因此解析时
    我们固定使用最后三个字段作为 angle/jam 标签，前面的部分全部视为类别名称。
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 256,
        center_crop: bool = True,
        random_flip: bool = False,
        metadata: Optional[SARDatasetMetadata] = None,
        angle_bin_size: Optional[float] = None,
    ) -> None:
        super().__init__()
        if not os.path.isdir(root_dir):
            raise ValueError(f"Provided root_dir={root_dir} is not a valid directory")

        self.root_dir = root_dir
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip

        # 收集所有 PNG 文件
        self.image_paths: List[str] = []
        for root, _dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    self.image_paths.append(os.path.join(root, file))

        if not self.image_paths:
            raise ValueError(f"No PNG files were found in directory: {root_dir}")

        # 解析文件名，建立映射
        if metadata is not None:
            # 如果传入 metadata，则直接使用已有映射，可保证 train/eval 一致
            self.class_to_id = dict(metadata.class_to_id)
            self.angle_to_id = dict(metadata.angle_to_id)
            self.jam_a_to_id = dict(metadata.jam_a_to_id)
            self.jam_p_to_id = dict(metadata.jam_p_to_id)
            if metadata.angle_bin_size is not None:
                self.angle_bin_size = float(metadata.angle_bin_size)
            else:
                self.angle_bin_size = float(angle_bin_size) if angle_bin_size is not None else None
        else:
            classes, angles, jam_as, jam_ps = self._scan_labels(self.image_paths)
            self.class_to_id = {name: idx for idx, name in enumerate(sorted(classes))}
            if angle_bin_size is not None:
                if angle_bin_size <= 0:
                    raise ValueError("angle_bin_size must be a positive number")
                self.angle_bin_size = float(angle_bin_size)
                num_bins = int(math.ceil(360.0 / self.angle_bin_size))
                self.angle_to_id = {}
                for name in angles:
                    bin_idx = self._angle_to_bin(name, num_bins)
                    self.angle_to_id[name] = bin_idx
            else:
                self.angle_bin_size = None
                # 角度这里默认按照数值排序；若无法转成 float，则回退到字符串排序
                try:
                    sorted_angles = sorted(angles, key=lambda v: float(v))
                except ValueError:
                    sorted_angles = sorted(angles)
                self.angle_to_id = {name: idx for idx, name in enumerate(sorted_angles)}
            self.jam_a_to_id = {name: idx for idx, name in enumerate(sorted(jam_as))}
            self.jam_p_to_id = {name: idx for idx, name in enumerate(sorted(jam_ps))}

        if metadata is None and angle_bin_size is None:
            self.angle_bin_size = None
        elif metadata is None and angle_bin_size is not None:
            # 已在上面赋值
            pass
        elif metadata is not None and metadata.angle_bin_size is None:
            self.angle_bin_size = float(angle_bin_size) if angle_bin_size is not None else None

        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        self.id_to_angle = {idx: name for name, idx in self.angle_to_id.items()}
        self.id_to_jam_a = {idx: name for name, idx in self.jam_a_to_id.items()}
        self.id_to_jam_p = {idx: name for name, idx in self.jam_p_to_id.items()}

        self.num_classes = len(self.class_to_id)
        if self.angle_bin_size is not None:
            self.num_angles = max(self.angle_to_id.values()) + 1 if self.angle_to_id else 0
        else:
            self.num_angles = len(self.angle_to_id)
        self.num_jam_a = len(self.jam_a_to_id)
        self.num_jam_p = len(self.jam_p_to_id)

        # torchvision transforms：灰度读取 -> Resize -> CenterCrop -> ToTensor -> 映射到 [-1, 1]
        interpolation = transforms.InterpolationMode.BILINEAR
        transform_list: List[transforms.Compose] = [transforms.Resize(image_size, interpolation=interpolation)]
        if center_crop:
            transform_list.append(transforms.CenterCrop(image_size))
        else:
            transform_list.append(transforms.RandomCrop(image_size))
        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transform_list)

    @staticmethod
    def _scan_labels(image_paths: List[str]) -> Tuple[set, set, set, set]:
        classes, angles, jam_as, jam_ps = set(), set(), set(), set()
        for path in image_paths:
            cls, angle, jam_a, jam_p = SARDataset.parse_filename(path)
            classes.add(cls)
            angles.add(angle)
            jam_as.add(jam_a)
            jam_ps.add(jam_p)
        return classes, angles, jam_as, jam_ps

    @staticmethod
    def parse_filename(path: str) -> Tuple[str, str, str, str]:
        """解析文件名获取四类条件标签。"""

        filename = os.path.basename(path)
        stem, _ext = os.path.splitext(filename)
        parts = stem.split("_")
        if len(parts) < 4:
            raise ValueError(
                f"Filename `{filename}` does not follow the expected pattern class_angle_jamA_jamP.png"
            )
        jam_p = parts[-1]
        jam_a = parts[-2]
        angle = parts[-3]
        class_name = "_".join(parts[:-3])
        if not class_name:
            raise ValueError(f"Filename `{filename}` has empty class name")
        return class_name, angle, jam_a, jam_p

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        class_name, angle, jam_a, jam_p = self.parse_filename(image_path)

        image = Image.open(image_path).convert("L")
        image = self.transform(image)
        # ToTensor 会输出 [0,1]，这里映射到 [-1,1]
        pixel_values = image * 2.0 - 1.0

        class_id = self.class_to_id[class_name]
        if angle in self.angle_to_id:
            angle_id = self.angle_to_id[angle]
        elif self.angle_bin_size is not None:
            # 对 eval 集中未出现过的新角度，根据相同的 bin 规则进行量化
            num_bins = int(math.ceil(360.0 / self.angle_bin_size))
            angle_id = self._angle_to_bin(angle, num_bins)
            self.angle_to_id[angle] = angle_id
            self.id_to_angle[angle_id] = angle
        else:
            raise KeyError(f"angle `{angle}` not found in dataset mapping")
        jam_a_id = self.jam_a_to_id[jam_a]
        jam_p_id = self.jam_p_to_id[jam_p]

        return {
            "pixel_values": pixel_values,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "angle_id": torch.tensor(angle_id, dtype=torch.long),
            "jam_a_id": torch.tensor(jam_a_id, dtype=torch.long),
            "jam_p_id": torch.tensor(jam_p_id, dtype=torch.long),
        }

    def get_metadata(self) -> SARDatasetMetadata:
        return SARDatasetMetadata(
            num_classes=self.num_classes,
            num_angles=self.num_angles,
            num_jam_a=self.num_jam_a,
            num_jam_p=self.num_jam_p,
            image_size=self.image_size,
            class_to_id=self.class_to_id,
            angle_to_id=self.angle_to_id,
            jam_a_to_id=self.jam_a_to_id,
            jam_p_to_id=self.jam_p_to_id,
            angle_bin_size=self.angle_bin_size,
        )

    def _angle_to_bin(self, angle_name: str, num_bins: int) -> int:
        """将角度字符串映射到等宽分箱后的 bin index。"""

        try:
            angle_value = float(angle_name)
        except ValueError:
            raise ValueError(
                "angle_bin_size 模式下要求角度能够转成 float, 当前角度无法解析: "
                f"{angle_name}"
            )
        # clip 到 [0, 360)
        angle_value = max(0.0, min(359.9999, angle_value))
        bin_idx = int(angle_value // self.angle_bin_size)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        return bin_idx

    def save_metadata(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.get_metadata().to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_metadata(cls, metadata_path: str) -> SARDatasetMetadata:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SARDatasetMetadata.from_dict(data)
