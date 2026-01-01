import functools
import json
import os
import time

import click
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import wandb
from cs336_basics.config import Configures, TrainConfig
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerBlock, TransformerLM
from cs336_basics.optimizer import AdamW, cross_entropy, get_lr_cosine_schedule

# NOTE: Removed gradient_clipping import as FSDP requires internal clipping
# from cs336_basics.optimizer import gradient_clipping


def is_master():
    return int(os.environ.get("RANK", 0)) == 0


def get_device():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


@torch.no_grad()
def valid(model: nn.Module, valid_set: np.ndarray, cfg: Configures, iters: int = 10, dtype: torch.dtype | None = None):
    model.eval()
    losses = torch.zeros(iters, device=cfg.model.device)

    for k in range(iters):
        x, y = get_batch(valid_set, cfg.train.batch_size, cfg.model.context_length, cfg.model.device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = cross_entropy(logits, y)
        losses[k] = loss

    mean_loss = losses.mean()

    # NOTE: FSDP/DDP requires all-reduce to get global validation loss
    if dist.is_initialized():
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)

    model.train()
    return mean_loss.item()


def train(
    model: FSDP,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    start_step: int,
    cfg: Configures,
    train_set: np.ndarray,
    valid_set: np.ndarray,
    dtype: torch.dtype,
    rank: int,
    local_rank: int,
):
    model.train()

    tc: TrainConfig = cfg.train
    t_start = time.perf_counter()

    for it in range(start_step, tc.steps):
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        lr = get_lr_cosine_schedule(it, tc.lr_max, tc.lr_min, tc.t_w, tc.t_c)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        accum_loss = 0.0

        for micro_step in range(tc.accum_steps):
            x, y = get_batch(train_set, tc.batch_size, cfg.model.context_length, cfg.model.device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = cross_entropy(logits, y) / tc.accum_steps

            accum_loss += loss.item()
            # NOTE: Scale loss for FP16 to prevent underflow
            scaler.scale(loss).backward()

        # NOTE: Unscale gradients before clipping
        scaler.unscale_(optimizer)
        total_norm = model.clip_grad_norm_(tc.grad_clip)
        # NOTE: scaler.step skips update if gradients contain inf/nan
        scaler.step(optimizer)
        scaler.update()

        if tc.save.enable and ((it + 1) % tc.save.interval == 0 or it == 0):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            tps = (
                tc.batch_size
                * cfg.model.context_length
                * tc.accum_steps
                * tc.save.interval
                * int(os.environ.get("WORLD_SIZE", 1))
            ) / dt

            vloss = valid(model, valid_set, cfg, dtype=dtype)

            # NOTE: FSDP state_dict_type is a collective op, all ranks must enter
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                raw_state_dict = model.state_dict()
                if rank == 0:
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    dt_interval = t1 - t_start
                    dt_step = t1 - t0
                    tps = (
                        tc.batch_size
                        * cfg.model.context_length
                        * tc.accum_steps
                        * tc.save.interval
                        * int(os.environ.get("WORLD_SIZE", 1))
                    ) / dt_interval

                    free_mem, total_mem = torch.cuda.mem_get_info(local_rank)
                    metrics = {
                        "step": it,
                        "time": dt_step,
                        "valid/loss": vloss,
                        "train/loss": accum_loss,
                        "perf/tps": tps,
                        "train/lr": lr,
                        "gpu/mem": (total_mem - free_mem) / 1024**2,
                        "perf/grad_norm": total_norm.item(),
                    }
                    wandb.log(metrics)
                    print(f"Step {it:04d} | Time (step): {dt_step:.4f} | Loss: {vloss:.4f} | TPS: {tps:.1f}")
                    checkpoint = {"model": raw_state_dict, "optimizer": optimizer.state_dict(), "iteration": it}
                    torch.save(checkpoint, f"./dist/checkpoint_{it:04d}.pt")
                # Reset interval timer after recording
                t_start = time.perf_counter()


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), required=True)
@click.option("-r", "--resume", type=click.Path(True), help="resume form a checkpoint file")
@click.option("-m", "--mmap", type=click.BOOL, default=False, help="use mmap load data to save memory")
@click.option("--fp16", type=click.BOOL, default=False, help="use fp16 and GradScaler to train")
def main(config, resume, mmap, fp16):
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = get_device()

    # Ensure output directory exists
    if rank == 0:
        os.makedirs("./dist", exist_ok=True)
    dist.barrier()  # Wait for rank 0 to create directory

    try:
        with open(config) as f:
            conf = json.load(f)
        cfg = Configures(**conf)
    except Exception as e:
        if rank == 0:
            print(f"check config: {e}")
        return

    cfg.model.device = str(device)

    if rank == 0:
        wandb.init(
            project="cs336-assignment1-basic-fsdp",
            name=f"run_FSDP_{world_size}x_d{cfg.model.d_model}",
            config=cfg.model_dump(),
            settings=wandb.Settings(quiet=True, silent=True),
        )

    # NOTE: Set seed but allow divergence in data sampling later
    torch.manual_seed(cfg.seed)

    # NOTE: Use FP16 with GradScaler for DCU Z100 if requested
    dtype = torch.float16 if fp16 else torch.float32

    if mmap:
        train_set = np.load(cfg.data.train, mmap_mode="r")
        valid_set = np.load(cfg.data.valid, mmap_mode="r")
    else:
        train_set = np.load(cfg.data.train)
        valid_set = np.load(cfg.data.valid)

    # Initialization on CPU (or specific device) if needed, but FSDP handles moving
    # Using 'meta' device or CPU is common for large models, but here we stick to standard flow
    model = TransformerLM(**cfg.model.model_dump())

    # NOTE: FSDP Auto Wrapping Policy ensures TransformerBlocks are sharded efficiently
    fsdp_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # NOTE: Wrap model with FSDP
    # cpu_offload=CPUOffload(offload_params=True) can be added if OOM occurs
    model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy, device_id=device)

    optimizer = AdamW(model.parameters(), **cfg.optimizer.model_dump())
    # NOTE: GradScaler for FP16 mixed precision training (disabled if fp16=False)
    scaler = GradScaler(enabled=fp16)

    start_step = 0
    if resume:
        # NOTE: Load full checkpoint requires FSDP/Torch specific handling
        # Simple approach: Load to CPU, let FSDP distribute (via load_state_dict)
        # However, requires state_dict to meet FSDP structure or process inside context
        # For simplicity in this homework scope, assuming standard load if format matches
        # or simplified reloading.
        if rank == 0:
            print("Warning: Resume logic in FSDP requires careful state dict handling.")
        pass

    # NOTE: Diverge RNG for data sampling
    torch.manual_seed(cfg.seed + rank)

    train(model, optimizer, scaler, start_step, cfg, train_set, valid_set, dtype, rank, local_rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
