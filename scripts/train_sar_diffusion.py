import argparse
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.sar_dataset import SARDataset, SARDatasetMetadata
from models.sar_unet import SarConditionalUNet


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional SAR diffusion model using accelerate")
    parser.add_argument("--train_data_dir", type=str, required=True, help="训练集根目录")
    parser.add_argument("--validation_data_dir", type=str, default=None, help="验证集根目录，可选")
    parser.add_argument("--output_dir", type=str, default="sar-diffusion")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # 数据相关
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", action="store_true", help="是否使用中心裁剪")
    parser.add_argument("--random_flip", action="store_true", help="是否随机左右翻转")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # 模型参数
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--embed_dim_each", type=int, default=128)
    parser.add_argument("--attention_head_dim", type=int, default=8)

    # 训练超参
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_ema", action="store_true")

    # 噪声调度器
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")

    # 日志与 checkpoint
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="每多少 step 保存一次模型")
    parser.add_argument("--checkpointing_epochs", type=int, default=None, help="每多少个 epoch 保存模型")
    parser.add_argument("--validation_epochs", type=int, default=1)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()
    return args


def create_dataloader(dataset: SARDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def save_checkpoint(
    accelerator: Accelerator,
    model: SarConditionalUNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    epoch: int,
    global_step: int,
    metadata: Dict,
    best_metric: Optional[float],
    output_dir: Path,
    is_best: bool = False,
) -> Path:
    checkpoint = {
        "model": accelerator.get_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "metadata": metadata,
        "model_config": model.config.to_dict(),
        "best_metric": best_metric,
    }

    scaler = getattr(accelerator, "scaler", None)
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"checkpoint-step-{global_step}.pt"
    save_path = checkpoints_dir / ckpt_name
    accelerator.save(checkpoint, save_path)
    if is_best:
        accelerator.save(checkpoint, checkpoints_dir / "best.pt")
    return save_path


def maybe_load_checkpoint(
    model: SarConditionalUNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    accelerator: Accelerator,
    resume_path: str,
):
    logger.info(f"Resuming from checkpoint: {resume_path}")
    map_location = accelerator.device
    checkpoint = torch.load(resume_path, map_location=map_location)
    accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    scaler = getattr(accelerator, "scaler", None)
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    best_metric = checkpoint.get("best_metric", None)
    metadata = checkpoint.get("metadata")
    return epoch, global_step, best_metric, metadata


def main():
    args = parse_args()

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=["tensorboard"],
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if accelerator.is_local_main_process:
        if os.path.exists(args.output_dir) and args.overwrite_output_dir:
            logger.warning("Overwriting output directory")
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )

    logger.info(accelerator.state)
    accelerator.init_trackers("sar_diffusion")
    set_seed(args.seed)

    # 构建数据集
    train_dataset = SARDataset(
        args.train_data_dir,
        image_size=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    train_metadata_obj = train_dataset.get_metadata()
    train_metadata = train_metadata_obj.to_dict()

    if accelerator.is_main_process:
        train_dataset.save_metadata(os.path.join(args.output_dir, "dataset_metadata.json"))

    eval_dataset = None
    if args.validation_data_dir is not None:
        eval_dataset = SARDataset(
            args.validation_data_dir,
            image_size=args.resolution,
            center_crop=args.center_crop,
            random_flip=False,
            metadata=train_metadata_obj,
        )

    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = create_dataloader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
        )

    # 构建模型
    model = SarConditionalUNet(
        image_size=args.resolution,
        in_channels=1,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        cond_dim=args.cond_dim,
        embed_dim_each=args.embed_dim_each,
        attention_head_dim=args.attention_head_dim,
        num_classes=train_dataset.num_classes,
        num_angles=train_dataset.num_angles,
        num_jam_a=train_dataset.num_jam_a,
        num_jam_p=train_dataset.num_jam_p,
    )

    if args.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()

    # Exponential Moving Average
    ema_model = EMAModel(model.parameters()) if args.use_ema else None

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None
        else args.max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num training epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = args.max_train_steps
        args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # 恢复 checkpoint
    starting_epoch = 0
    global_step = 0
    best_metric = None
    if args.resume_from_checkpoint is not None:
        starting_epoch, global_step, best_metric, resume_metadata = maybe_load_checkpoint(
            model, optimizer, lr_scheduler, accelerator, args.resume_from_checkpoint
        )
        if resume_metadata is not None:
            dataset_meta_dict = resume_metadata.get("dataset")
            if dataset_meta_dict is not None:
                train_metadata_obj = SARDatasetMetadata.from_dict(dataset_meta_dict)
                train_metadata = train_metadata_obj.to_dict()

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                clean_images = batch["pixel_values"]
                class_id = batch["class_id"]
                angle_id = batch["angle_id"]
                jam_a_id = batch["jam_a_id"]
                jam_p_id = batch["jam_p_id"]

                noise = torch.randn_like(clean_images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean_images.shape[0],),
                    device=clean_images.device,
                    dtype=torch.long,
                )
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                noise_pred = model(noisy_images, timesteps, class_id, angle_id, jam_a_id, jam_p_id)
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if ema_model is not None:
                    ema_model.step(model.parameters())

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if args.checkpointing_steps is not None and global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        global_step,
                        metadata={"dataset": train_metadata, "model": model.config.to_dict()},
                        best_metric=best_metric,
                        output_dir=Path(args.output_dir),
                    )

            if global_step >= max_train_steps:
                break

            if accelerator.is_main_process and global_step % args.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} Step {step} Global {global_step} Loss {loss.detach().item():.4f} LR {lr_scheduler.get_last_lr()[0]:.6f}"
                )

        if accelerator.is_main_process and args.checkpointing_epochs is not None:
            if (epoch + 1) % args.checkpointing_epochs == 0:
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    metadata={"dataset": train_metadata, "model": model.config.to_dict()},
                    best_metric=best_metric,
                    output_dir=Path(args.output_dir),
                )

        # 验证
        if eval_dataloader is not None and (epoch + 1) % args.validation_epochs == 0:
            model.eval()
            losses = []
            for batch in eval_dataloader:
                with torch.no_grad():
                    clean_images = batch["pixel_values"]
                    class_id = batch["class_id"]
                    angle_id = batch["angle_id"]
                    jam_a_id = batch["jam_a_id"]
                    jam_p_id = batch["jam_p_id"]

                    noise = torch.randn_like(clean_images)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (clean_images.shape[0],),
                        device=clean_images.device,
                        dtype=torch.long,
                    )
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                    noise_pred = model(noisy_images, timesteps, class_id, angle_id, jam_a_id, jam_p_id)
                    val_loss = F.mse_loss(noise_pred, noise)
                    losses.append(accelerator.gather(val_loss.repeat(clean_images.shape[0])).mean().item())

            mean_loss = sum(losses) / len(losses)
            if accelerator.is_main_process:
                logger.info(f"Eval epoch {epoch}: loss={mean_loss:.4f}")
                accelerator.log({"eval_loss": mean_loss}, step=global_step)
                improved = best_metric is None or mean_loss < best_metric
                if improved:
                    best_metric = mean_loss
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        global_step,
                        metadata={"dataset": train_metadata, "model": model.config.to_dict()},
                        best_metric=best_metric,
                        output_dir=Path(args.output_dir),
                        is_best=True,
                    )

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_save_path = save_checkpoint(
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            epoch=args.num_train_epochs - 1,
            global_step=global_step,
            metadata={"dataset": train_metadata, "model": model.config.to_dict()},
            best_metric=best_metric,
            output_dir=Path(args.output_dir),
        )
        logger.info(f"Training complete. Final checkpoint saved to {final_save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
