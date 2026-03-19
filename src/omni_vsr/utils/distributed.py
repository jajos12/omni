"""Distributed training helpers."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = "nccl") -> DistributedContext:
    if backend == "auto":
        if platform.system() == "Windows":
            backend = "gloo"
        else:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedContext(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)
    return DistributedContext(
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device(f"cuda:{local_rank}"),
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
