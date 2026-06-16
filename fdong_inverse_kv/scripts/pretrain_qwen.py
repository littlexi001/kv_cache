import argparse
import os

import torch
import torch.distributed as dist

from train_common import add_training_arguments, train


def main():
    parser = argparse.ArgumentParser(description="Distributed inverse-KV Qwen3 pretraining")
    add_training_arguments(parser)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    try:
        train(local_rank, world_size, torch.device(f"cuda:{local_rank}"), args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
