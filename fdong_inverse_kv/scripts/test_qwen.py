import os
import time
import torch
import pickle
import argparse
import itertools
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import DeepSeekDistillation, TokenizedJSONLData
from models import MyQwen3ForCausalLM, ActiveLearningFilter
from transformers import AutoTokenizer, AutoConfig, get_cosine_schedule_with_warmup, AutoModelForCausalLM
from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

@torch.no_grad()
def prepare_model(local_rank, world_size, device, args):
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code = True)
    config.use_moe = args.use_moe
    config.moe_intermediate_size = args.moe_intermediate_size
    config.num_experts_per_tok = args.expert_per_token
    config.num_experts = args.num_experts
    config.gating_reference = args.gating_reference
    config.norm_topk_prob = True

    config.enable_active_learning = False
    config.active_learning_k = 0
    config.active_learning_percent = 0
    config.active_learning_from = 0

    model_class = MyQwen3ForCausalLM
    model = model_class(config).to(device)

    # model = DDP(model, device_ids=[device], find_unused_parameters=args.use_moe)
    
    print(f'rank {local_rank} model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(local_rank, world_size, args):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer)

    print(f"Construct dataset, total {len(dataset)} samples.")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=args.data_shuffle)
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers, sampler=sampler)

    return dataloader


def prepare_loss(s_model):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643, reduction='mean')
    scaler = torch.amp.GradScaler('cuda')
    
    return token_loss_fn,  scaler


def forward_step(local_rank, device, source, target, model, token_loss_fn, args):
    source, target = source.to(device), target.to(device)

    s_output = model(source, output_hidden_states = False)
    s_logits = s_output.logits
    
    loss = token_loss_fn(s_logits.view(-1, s_logits.size(-1)), target.reshape(-1))

    return loss


def cal_test_loss(local_rank, world_size, device, model, dataloader, token_loss_fn, args):
    losses = []
    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda', enabled=args.use_bf16):
            loss = forward_step(local_rank, device, source, target, model, token_loss_fn, args)
        
        if world_size > 1:
            local_loss_tensor = torch.tensor(computed_loss, dtype=torch.float32, device=device_id)
            dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
            global_sum_loss = local_loss_tensor.item() # 因为是标量，可以直接取 item()
            loss = global_sum_loss / world_size

        losses.append(loss.item())
        if local_batch_idx >= args.test_batch_size // args.local_batch_size:
            break

    loss = sum(losses) / len(losses)
    return loss


def draw_test_loss(test_loss_dict = None, trials = None):
    if test_loss_dict is None:
        test_loss_dict = pickle.load(open("../figures/test_loss.pkl", "rb"))
    if trials is None:
        trials = {
            "al-k500-drop50%-from10000": "../checkpoints/baseline-256-AL-500-0.5-10000/",
            "al-k500-drop50%-from20000": "../checkpoints/baseline-256-AL-500-0.5-20000/",
            "al-k500-drop25%-from10000": "../checkpoints/baseline-256-AL-500-0.75-10000/",
            "al-k500-drop25%-from20000": "../checkpoints/baseline-256-AL-500-0.75-20000/",
            "al-k800-drop50%-from10000": "../checkpoints/baseline-256-AL-800-0.5-10000/",
            "al-k800-drop50%-from20000": "../checkpoints/baseline-256-AL-800-0.5-20000/",
            "al-k800-drop25%-from10000": "../checkpoints/baseline-256-AL-800-0.75-10000/",
            "al-k800-drop25%-from20000": "../checkpoints/baseline-256-AL-800-0.75-20000/",
            "baseline-batchsize-128": "../checkpoints/baseline-128/",
            "baseline-batchsize-256": "../checkpoints/baseline-256-AL-200-1.0-5000/",
        }
    fig = plt.figure(figsize=(10, 5))
    for trial, loss_dict in test_loss_dict.items():
        if trial not in trials:
            continue
        plt.plot(loss_dict.keys(), loss_dict.values(), label=trial)
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs. Iteration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('../figures/test-loss.png')


@torch.no_grad()
def thread_main(local_rank, world_size, device, args):
    print(f"running on device {local_rank}")
    # if local_rank == 0:
    #     if not os.path.exists(args.ckpt_dir):
    #         os.makedirs(args.ckpt_dir)

    dataloader = prepare_data(local_rank, world_size, args)
    model = prepare_model(local_rank, world_size, device, args)
    token_loss_fn, scaler = prepare_loss(model)

    trials = {
        "tokenfreq-adam": "../checkpoints/baseline-256-adamw-8e-5-TOKEN_FREQ_NORM/",
        "baseline-128": "../checkpoints/baseline-128/",
        "baseline-256": "../checkpoints/baseline-256/",
    }

    steps = [s for s in range(2000, 20000, 2000)]
    steps += [s for s in range(20000, 200000, 5000)]
    steps += [s for s in range(200000, 1000000, 10000)]
    if os.path.exists("../figures/test_loss.pkl"):
        test_loss = pickle.load(open("../figures/test_loss.pkl", "rb"))
    else:
        test_loss = {}
    for trial, ckpt_dir in trials.items():
        if trial not in test_loss:
            test_loss[trial] = {}
        for step in steps:
            if step in test_loss[trial]:
                print(f"{trial} step: skip step {step}")
                continue
            ckpt_file = os.path.join(ckpt_dir, f"{step}.pth")
            if not os.path.exists(ckpt_file):
                print(f"{trial} stop at step {step}")
                break
            print(f"{trial} step: {step}")
            model.load_state_dict(torch.load(ckpt_file, weights_only=True))
            loss = cal_test_loss(local_rank, world_size, device, model, dataloader, token_loss_fn, args)
            test_loss[trial][step] = loss
            print(f"{trial} step: {step} loss: {loss}")
    pickle.dump(test_loss, open("../figures/test_loss.pkl", "wb"))
    draw_test_loss(test_loss, trials)
            

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Batch & training config
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")

    parser.add_argument("--use_moe", action="store_true", default=False)
    parser.add_argument("--no_use_moe", action="store_false", dest="use_moe")
    parser.add_argument("--moe_intermediate_size", type=int, default=1536)
    parser.add_argument("--expert_per_token", type=int, default=2)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--gating_reference", type=str, choices=["oracle", "switch"], default="switch")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", type=str, default="../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--test_batch", type=int, default=0)

    args = parser.parse_args()

    return args




def main():
    # dist.init_process_group(backend="nccl")
    local_rank = 0 # int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = 1 #int(os.environ["WORLD_SIZE"])

    args = parse_args()

    if local_rank == 0:
        print("Training Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    print(f"local_rank: {local_rank}, world_size: {world_size}")

    thread_main(local_rank, world_size, device, args)


if __name__ == "__main__":
    main()
    # draw_test_loss()

