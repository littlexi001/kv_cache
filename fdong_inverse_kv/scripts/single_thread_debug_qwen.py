import argparse

import torch

from train_common import add_training_arguments, train


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Single-process inverse-KV Qwen3 debug training")
    add_training_arguments(parser)
    args = parser.parse_args()
    train(0, 1, select_device(), args)


if __name__ == "__main__":
    main()
