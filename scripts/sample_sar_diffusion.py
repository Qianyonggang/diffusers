import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from torchvision.utils import make_grid, save_image

from data.sar_dataset import SARDataset, SARDatasetMetadata
from models.sar_unet import SarConditionalUNet, SarUNetConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Sample SAR images from a conditional diffusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="sar-diffusion-samples")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--scheduler", type=str, choices=["ddim", "dpm"], default="dpm")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--class_name", type=str, required=True, help="条件：类别字符串")
    parser.add_argument("--angle", type=str, required=True, help="条件：角度（与训练数据文件名中的字符串一致）")
    parser.add_argument("--jam_active", type=int, choices=[0, 1], required=True, help="条件：是否有源干扰")
    parser.add_argument("--jam_passive", type=int, choices=[0, 1], required=True, help="条件：是否无源干扰")

    parser.add_argument("--metadata_path", type=str, default=None, help="可选：显式指定数据集 metadata json")

    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = checkpoint.get("metadata")
    if metadata is None:
        raise ValueError("Checkpoint missing metadata; please provide --metadata_path")
    dataset_meta_dict = metadata.get("dataset")
    model_meta_dict = metadata.get("model")
    if dataset_meta_dict is None or model_meta_dict is None:
        raise ValueError("Checkpoint metadata is incomplete")

    dataset_meta = SARDatasetMetadata.from_dict(dataset_meta_dict)
    model_config = SarUNetConfig.from_dict(model_meta_dict)

    model = SarConditionalUNet.from_config(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, dataset_meta


def prepare_condition_tensors(meta: SARDatasetMetadata, args) -> Dict[str, torch.Tensor]:
    if args.class_name not in meta.class_to_id:
        raise ValueError(f"class_name `{args.class_name}` not found in metadata; available: {list(meta.class_to_id.keys())}")

    angle_key = str(args.angle)
    if angle_key not in meta.angle_to_id:
        raise ValueError(f"angle `{angle_key}` not in metadata; available: {list(meta.angle_to_id.keys())}")

    jam_a_key = str(args.jam_active)
    jam_p_key = str(args.jam_passive)
    if jam_a_key not in meta.jam_a_to_id or jam_p_key not in meta.jam_p_to_id:
        raise ValueError("jam_active or jam_passive value not found in metadata")

    class_ids = torch.full((args.num_samples,), meta.class_to_id[args.class_name], dtype=torch.long)
    angle_ids = torch.full((args.num_samples,), meta.angle_to_id[angle_key], dtype=torch.long)
    jam_a_ids = torch.full((args.num_samples,), meta.jam_a_to_id[jam_a_key], dtype=torch.long)
    jam_p_ids = torch.full((args.num_samples,), meta.jam_p_to_id[jam_p_key], dtype=torch.long)

    return {
        "class_id": class_ids,
        "angle_id": angle_ids,
        "jam_a_id": jam_a_ids,
        "jam_p_id": jam_p_ids,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优先从 checkpoint 中读取 metadata；如果缺失则尝试外部 json
    try:
        model, dataset_meta = load_checkpoint(args.checkpoint, device)
    except ValueError:
        if args.metadata_path is None:
            raise
        dataset_meta = SARDataset.load_metadata(args.metadata_path)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model_config = SarUNetConfig.from_dict(checkpoint["model_config"])
        model = SarConditionalUNet.from_config(model_config)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

    scheduler_cls = DPMSolverMultistepScheduler if args.scheduler == "dpm" else DDIMScheduler
    scheduler = scheduler_cls(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    scheduler.set_timesteps(args.num_inference_steps, device=device)

    cond_tensors = prepare_condition_tensors(dataset_meta, args)
    for key in cond_tensors:
        cond_tensors[key] = cond_tensors[key].to(device)

    noise = torch.randn((args.num_samples, 1, dataset_meta.image_size, dataset_meta.image_size), device=device)
    model.eval()
    with torch.no_grad():
        sample = noise
        for t in scheduler.timesteps:
            timestep = torch.full((args.num_samples,), t, device=device, dtype=torch.long)
            noise_pred = model(
                sample,
                timestep,
                cond_tensors["class_id"],
                cond_tensors["angle_id"],
                cond_tensors["jam_a_id"],
                cond_tensors["jam_p_id"],
            )
            step_output = scheduler.step(noise_pred, t, sample)
            sample = step_output.prev_sample

    # 把 [-1,1] 映射回 [0,1]
    images = (sample.clamp(-1, 1) + 1) / 2.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 单张保存
    base_name = f"{args.class_name}_{args.angle}_{args.jam_active}_{args.jam_passive}"
    for idx, image in enumerate(images):
        save_path = output_dir / f"{base_name}_{idx:02d}.png"
        save_image(image, save_path)

    grid = make_grid(images, nrow=min(args.num_samples, 4))
    save_image(grid, output_dir / f"{base_name}_grid.png")
    print(f"Saved samples to {output_dir}")


if __name__ == "__main__":
    main()
