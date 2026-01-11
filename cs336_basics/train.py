import json
import time
from collections.abc import Iterator
from typing import cast

import click
import numpy as np
import torch
from torch import nn, optim

import wandb
from cs336_basics.config import Configures, TrainConfig, update_cfg_w_sweep
from cs336_basics.data import get_batch_iterator, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import (
    AdamW,
    Muon,
    cross_entropy,
    get_lr_cosine_schedule,
    get_lr_wsd_schedule,
    gradient_clipping,
)


@torch.no_grad()
def valid(model: nn.Module, valid_iter: Iterator, cfg: Configures, iters: int = 10, dtype: torch.dtype | None = None):
    """
    get 10 sample from valid set and calculate the mean loss
    """
    # change model work mode to valid/inference
    model.eval()
    losses = torch.zeros(iters)

    for k in range(iters):
        x, y = next(valid_iter)

        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses[k] = loss.detach()

    # change back to train mode
    model.train()
    return losses.mean().item()


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    start_step: int,
    cfg: Configures,
    scaler: torch.GradScaler,
    train_set: np.ndarray,
    valid_set: np.ndarray,
    dtype: torch.dtype,
):
    # set the module in training mode
    model.train()

    tc: TrainConfig = cfg.train
    device = cfg.model.device

    # data iterator
    train_iter = get_batch_iterator(train_set, tc.batch_size, cfg.model.context_length, device)
    valid_iter = get_batch_iterator(valid_set, tc.batch_size, cfg.model.context_length, device)

    t0 = time.perf_counter()
    for it in range(start_step, tc.steps):
        # x, y = get_batch(train_set, tc.batch_size, cfg.model.context_length, device)
        optimizer.zero_grad(set_to_none=True)

        # use warm up + cosin lr dency schedule
        if tc.lr_scheduler == "cosine":
            lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
        if tc.lr_scheduler == "wsd":
            lr = get_lr_wsd_schedule(it, tc.lr_max, tc.lr_min, tc.steps, tc.t_w, tc.t_c)
        # update lr in all optimizer params groups
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # use gradient accumulation to simulate a larger batch size
        accum_loss = torch.zeros(1, device=device)
        for micro_step in range(tc.accum_steps):
            x, y = next(train_iter)

            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y) / tc.accum_steps
            accum_loss += loss.detach()

            # Since PyTorch sums gradients in the .grad attribute by default, calling backward on each
            # micro-batch loss (scaled by 1/accum_steps) computes the average gradient for the full batch size.
            # scale the loss to prevent gradient underflow when using mixed precision (fp16).
            scaler.scale(loss).backward()

        # unscale before clipping to ensure the clipping threshold is applied to the actual gradient values.
        scaler.unscale_(optimizer)
        # clip the gradients to prevent them from exploding
        total_norm = gradient_clipping(model.parameters(), tc.grad_clip)
        # do optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        # log and save checkpoint after each interval steps
        if tc.save.enable and ((it + 1) % tc.save.interval == 0 or it == 0):
            # Wait for all kernels in all streams on a CUDA device to complete, then count time.
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            tps = (tc.batch_size * cfg.model.context_length * tc.accum_steps * tc.save.interval) / dt

            # calc valid loss
            vloss = valid(model, valid_iter, cfg, dtype=dtype)
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            # log to wandb
            accum_loss_ = accum_loss.item()
            metrics = {
                "step": it,
                "time": dt,
                "valid/loss": vloss,
                "train/loss": accum_loss_,
                "perf/tps": tps,
                "train/lr": lr,
                "gpu/mem": (total_mem - free_mem) / 1024**2,
                "perf/grad_norm": total_norm,
                "samples": it * tc.batch_size * tc.accum_steps
            }
            wandb.log(metrics)
            print(
                f"Step {it:04d} | Time: {dt:.6f} | Valid Loss: {vloss:.4f} | "
                f"Train Loss: {accum_loss_:.4f} | TPS: {tps:.1f} | LR: {lr:.2e}"
            )

            # save_checkpoint(model, optimizer, it, f"./dist/checkpoint_{it:04d}_{vloss:.4f}.pt")
            t0 = time.perf_counter()


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True)
@click.option("--fp16", is_flag=True, default=False, help="enable fp16+grad_scaler train")
@click.option("-r", "--resume", type=click.Path(True), help="resume form a checkpoint file")
def main(config, fp16, resume):
    try:
        with open(config) as f:
            conf = json.load(f)
        cfg = Configures(**conf)
    except Exception as e:
        print(f"check config: {e}")
        return

    # init wandb
    wandb.init(
        project="cs336-assignment1-basic",
        group="TinyStories-stage1-lr2",
        config=cfg.model_dump(),
        reinit="finish_previous",
        settings=wandb.Settings(quiet=True, silent=True),
    )

    # wandb sweep
    if wandb.run is not None and wandb.run.sweep_id is not None:
        cfg = update_cfg_w_sweep(cfg, dict(wandb.config))
        run_name = (
            f"run_d{cfg.model.d_model}F{cfg.model.d_ff}L{cfg.model.num_layers}H{cfg.model.num_heads}"
            f"B{cfg.train.batch_size}*{cfg.train.accum_steps}LR{cfg.train.lr_max}-{cfg.train.lr_min}"
            f"_TinyStories"
        )

        wandb.run.name = run_name
        wandb.config.update(cfg.model_dump(), allow_val_change=True)

    # seed
    torch.manual_seed(cfg.seed)
    # precision
    if fp16:
        dtype = torch.float16
    elif torch.cuda.get_device_capability() >= (8, 0):
        dtype = torch.bfloat16
        torch.set_float32_matmul_precision("high")
    else:
        dtype = torch.float32

    # to scale gradients and prevent underflow during float16 mixed precision training
    scaler = torch.GradScaler(cfg.model.device, enabled=(dtype == torch.float16))
    # lazy load file
    train_set = np.load(cfg.data.train)
    valid_set = np.load(cfg.data.valid)
    # init Model
    model = TransformerLM(**cfg.model.model_dump())
    model.to(cfg.model.device)
    # init Optimizer
    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    if cfg.optimizer.type == "muon":
        optimizer = Muon(model, **cfg.optimizer.model_dump())

    # load from checkpoint
    start_step = 0
    if resume:
        start_step = load_checkpoint(resume, model, optimizer)

    # compile the model to speedup
    model = cast(nn.Module, torch.compile(model))

    # train
    train(model, optimizer, start_step, cfg, scaler, train_set, valid_set, dtype)


if __name__ == "__main__":
    main()
