import os
import time
import torch
import argparse
import torch.nn as nn

from utils import DeepSeekDistillation
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM
from torch.utils.data import DataLoader


@torch.no_grad()
def prepare_model(local_rank, world_size, device, args):
    # config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code = True)
    # model = Qwen3ForCausalLM(config).to(device)

    model = AutoModelForCausalLM.from_pretrained(args.config_dir, trust_remote_code=True)

    print(f'rank {local_rank} model ok, params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9:.2f}B/{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B') # 
    return model


def prepare_data(local_rank, world_size, args):
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir, trust_remote_code=True)
    dataset = TokenizedJSONLData(args.data_dir, args.seq_len, tokenizer)

    print(f"Construct dataset, total {len(dataset)} samples.")
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, num_workers=args.num_workers)

    return dataloader


def prepare_loss_optimizer(model, args):
    token_loss_fn = nn.CrossEntropyLoss(ignore_index=151643)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, 1000000)
    scaler = torch.amp.GradScaler('cuda')

    return token_loss_fn, optimizer, lr_scheduler, scaler


def forward_step(local_rank, device, source, target, model, token_loss_fn, args):
    source, target = source.to(device), target.to(device)

    output = model(source, output_hidden_states = False,)

    target = target.reshape(-1)
    loss = token_loss_fn(output.logits.view(-1, output.logits.size(-1)), target)

    return loss

def thread_main(local_rank, world_size, device, args):
    print(f"running on device {local_rank}")
    if local_rank == 0:
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

    dataloader = prepare_data(local_rank, world_size, args)
    model = prepare_model(local_rank, world_size, device, args)
    token_loss_fn, optimizer, lr_scheduler, scaler = prepare_loss_optimizer(model, args)

    gradient_accumulation_steps = args.global_batch_size // args.local_batch_size

    for local_batch_idx, (source, target, real_lens) in enumerate(dataloader, 1):
        global_batch_idx = local_batch_idx // gradient_accumulation_steps
        with torch.amp.autocast(dtype=torch.bfloat16, device_type='cuda', enabled=args.use_bf16):
            loss = forward_step(local_rank, device, source, target, model, token_loss_fn, args)
        scaler.scale(loss / gradient_accumulation_steps).backward()

        if local_batch_idx % gradient_accumulation_steps == 0:
            pass
            # gradient
            model.zero_grad()
            
        batch_time = time.time() - start_time
        if local_rank == 0:
            print(f"batch: {global_batch_idx}-{local_batch_idx}, loss: {loss:.3f}, batch_time: {batch_time:.3f}", flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # Batch & training config
    parser.add_argument("--local_batch_size", type=int, default=16)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=1024)

    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--optimizer", type=str, choices=["AdamW", "sgd"], default="AdamW")

    parser.add_argument("--data_shuffle", action="store_true", default=True)
    parser.add_argument("--no_data_shuffle", action="store_false", dest="data_shuffle")

    parser.add_argument("--use_bf16", action="store_true", default=True)
    parser.add_argument("--no_use_bf16", action="store_false", dest="use_bf16")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--config_dir", type=str, default="../../Qwen3-0.6B")
    parser.add_argument("--data_dir", type=str, default="../../dclm/global-shard_01_of_10")
    parser.add_argument("--ckpt_dir", type=str, default="")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    print("Training Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    print(f"local_rank: {0}, world_size: {1}")

    thread_main(0, 1, torch.device("cuda:0"), args)


if __name__ == "__main__":
    main()
    
