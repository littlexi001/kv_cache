import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from models import MyQwen3ForCausalLM
from utils import HierarchicalPatternData


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze frequency-width training dynamics.")
    parser.add_argument("--config_dir", type=str, default="../Qwen3-0.6B")
    parser.add_argument("--checkpoint_root", type=str, default="../checkpoints")
    parser.add_argument("--run_specs", type=str, required=True, help="alias:run_name:train_distribution[,..]")
    parser.add_argument("--output_path", type=str, default="../experiments/frequency-width-dynamics-analysis.json")
    parser.add_argument("--modes", type=str, default="learning_curve,gradients,svd,probe,lm_bias")
    parser.add_argument("--checkpoint_steps", type=str, default="all", help="'all' or comma-separated steps")

    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--synthetic_block_size", type=int, default=4)
    parser.add_argument("--synthetic_num_hierarchy_layers", type=int, default=2)
    parser.add_argument("--synthetic_content_token_count", type=int, default=256)
    parser.add_argument("--synthetic_num_units_per_layer", type=int, default=64)
    parser.add_argument("--synthetic_seed", type=int, default=0)
    parser.add_argument("--synthetic_min_token_id", type=int, default=1)
    parser.add_argument("--synthetic_zipf_alpha", type=float, default=1.3)
    parser.add_argument("--synthetic_zipf_shuffle_ranks", action="store_true", default=True)
    parser.add_argument("--synthetic_no_zipf_shuffle_ranks", action="store_false", dest="synthetic_zipf_shuffle_ranks")
    parser.add_argument("--eval_sampling_distribution", choices=["uniform", "zipf"], default="uniform")
    parser.add_argument("--frequency_feature_layer", type=int, default=1)
    parser.add_argument("--head_fraction", type=float, default=0.2)
    parser.add_argument("--tail_fraction", type=float, default=0.4)

    parser.add_argument("--gradient_samples", type=int, default=128)
    parser.add_argument("--gradient_batch_size", type=int, default=16)
    parser.add_argument("--gradient_param_regex", type=str, default="")
    parser.add_argument("--svd_samples", type=int, default=256)
    parser.add_argument("--svd_top_dims", type=str, default="1,2,5,10,20")
    parser.add_argument("--probe_train_fraction", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=123)
    return parser.parse_args()


def parse_run_specs(text):
    specs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("Each run spec must be alias:run_name:train_distribution")
        alias, run_name, distribution = parts
        if distribution not in ("uniform", "zipf"):
            raise ValueError("train_distribution must be uniform or zipf")
        specs.append({"alias": alias, "run_name": run_name, "train_distribution": distribution})
    return specs


def parse_modes(text):
    return {item.strip() for item in text.split(",") if item.strip()}


def parse_ints(text):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_checkpoints(run_dir, checkpoint_steps):
    wanted = None if checkpoint_steps == "all" else set(parse_ints(checkpoint_steps))
    items = []
    if not os.path.isdir(run_dir):
        return items
    for name in os.listdir(run_dir):
        if not name.endswith(".pth"):
            continue
        stem = name[:-4]
        if not stem.isdigit():
            continue
        step = int(stem)
        if wanted is not None and step not in wanted:
            continue
        items.append((step, os.path.join(run_dir, name)))
    return sorted(items)


def load_runtime_config(run_dir):
    path = os.path.join(run_dir, "runtime_config.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_debug_config_from_state(config, state):
    embed = state["model.embed_tokens.weight"]
    config.vocab_size = int(embed.shape[0])
    config.hidden_size = int(embed.shape[1])
    layer_ids = []
    for key in state:
        match = re.match(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_ids.append(int(match.group(1)))
    config.num_hidden_layers = max(layer_ids) + 1

    q_proj = state["model.layers.0.self_attn.q_proj.weight"]
    k_proj = state["model.layers.0.self_attn.k_proj.weight"]
    q_norm = state["model.layers.0.self_attn.q_norm.weight"]
    config.head_dim = int(q_norm.shape[0])
    config.num_attention_heads = int(q_proj.shape[0] // config.head_dim)
    config.num_key_value_heads = int(k_proj.shape[0] // config.head_dim)
    if "model.layers.0.mlp.gate_proj.weight" in state:
        config.intermediate_size = int(state["model.layers.0.mlp.gate_proj.weight"].shape[0])
    config.max_position_embeddings = max(int(getattr(config, "max_position_embeddings", 0)), 256)
    if getattr(config, "pad_token_id", None) is None or config.pad_token_id >= config.vocab_size:
        config.pad_token_id = 0
    if getattr(config, "bos_token_id", None) is not None and config.bos_token_id >= config.vocab_size:
        config.bos_token_id = 1
    if getattr(config, "eos_token_id", None) is not None and config.eos_token_id >= config.vocab_size:
        config.eos_token_id = 2


def apply_runtime_config(config, runtime_config):
    config._attn_implementation = "eager"
    config.use_cache = False
    config.attention_stride_pattern = runtime_config.get("attention_stride_pattern", [1] * config.num_hidden_layers)
    config.residual_source_pattern = runtime_config.get("residual_source_pattern", [-1] * config.num_hidden_layers)
    config.use_moe = bool(runtime_config.get("use_moe", False))
    config.moe_num_unique_experts = int(runtime_config.get("moe_num_unique_experts", 4))
    config.moe_num_experts_per_tok = int(runtime_config.get("moe_num_experts_per_tok", 1))
    config.moe_intermediate_size = int(runtime_config.get("moe_intermediate_size", config.intermediate_size))
    config.moe_use_common_expert = bool(runtime_config.get("moe_use_common_expert", False))
    config.moe_common_intermediate_size = int(runtime_config.get("moe_common_intermediate_size", config.moe_intermediate_size))
    config.moe_router_bias = bool(runtime_config.get("moe_router_bias", False))
    config.moe_router_type = str(runtime_config.get("moe_router_type", "linear"))
    config.moe_router_hidden_size = int(runtime_config.get("moe_router_hidden_size", config.hidden_size))
    config.moe_router_act = str(runtime_config.get("moe_router_act", "silu"))
    config.moe_normalize_topk_prob = bool(runtime_config.get("moe_normalize_topk_prob", True))
    config.moe_router_input = str(runtime_config.get("moe_router_input", "hidden"))
    config.moe_router_input_pos = str(runtime_config.get("moe_router_input_pos", config.moe_router_input))
    config.moe_router_input_shape = str(runtime_config.get("moe_router_input_shape", "full"))
    config.moe_expert_input_pos = str(runtime_config.get("moe_expert_input_pos", "attention_output_residual"))
    config.moe_expert_input_shape = str(runtime_config.get("moe_expert_input_shape", "full"))
    config.moe_head_level = bool(runtime_config.get("moe_head_level", False))
    config.use_pre_router = bool(runtime_config.get("use_pre_router", False))
    config.pre_router_input = str(runtime_config.get("pre_router_input", "layer_input"))
    config.pre_router_controls_attention = bool(runtime_config.get("pre_router_controls_attention", False))


def load_model(args, run_dir, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location="cpu")
    runtime_config = load_runtime_config(run_dir)
    config = AutoConfig.from_pretrained(args.config_dir, trust_remote_code=True)
    infer_debug_config_from_state(config, state)
    apply_runtime_config(config, runtime_config)
    model = MyQwen3ForCausalLM(config).to(device).eval()
    model.load_state_dict(state, strict=True)
    return model, config


def make_dataset(args, sampling_distribution, num_samples=None):
    return HierarchicalPatternData(
        max_seq_len=args.seq_len,
        num_samples=args.num_samples if num_samples is None else num_samples,
        block_size=args.synthetic_block_size,
        num_hierarchy_layers=args.synthetic_num_hierarchy_layers,
        content_token_count=args.synthetic_content_token_count,
        num_units_per_layer=args.synthetic_num_units_per_layer,
        seed=args.synthetic_seed,
        min_token_id=args.synthetic_min_token_id,
        sampling_distribution=sampling_distribution,
        zipf_alpha=args.synthetic_zipf_alpha,
        zipf_shuffle_ranks=args.synthetic_zipf_shuffle_ranks,
        return_metadata=True,
    )


def frequency_scores(args, train_distribution):
    dataset = make_dataset(args, train_distribution, num_samples=1)
    if dataset.top_unit_sample_weights is None:
        return torch.ones(args.synthetic_num_units_per_layer, dtype=torch.float64)
    return torch.tensor(dataset.top_unit_sample_weights, dtype=torch.float64)


def make_bucket_ids(args, train_distribution):
    scores = frequency_scores(args, train_distribution)
    num_units = int(scores.numel())
    head_n = max(1, int(round(num_units * args.head_fraction)))
    tail_n = max(1, int(round(num_units * args.tail_fraction)))
    if head_n + tail_n >= num_units:
        raise ValueError("head_fraction + tail_fraction must leave at least one middle unit")
    sorted_ids = torch.argsort(scores, descending=True)
    bucket_ids = torch.full((num_units,), 1, dtype=torch.long)
    bucket_ids[sorted_ids[:head_n]] = 0
    bucket_ids[sorted_ids[-tail_n:]] = 2
    bucket_meta = {}
    for idx, name in enumerate(["head", "middle", "tail"]):
        ids = torch.nonzero(bucket_ids == idx, as_tuple=False).flatten()
        bucket_meta[name] = {
            "num_units": int(ids.numel()),
            "unit_ids": [int(x) for x in ids.tolist()],
            "score_sum": float(scores[ids].sum().item()),
            "score_mean": float(scores[ids].mean().item()),
        }
    return bucket_ids, bucket_meta


def batch_from_dataset(dataset, start, end):
    items = [dataset[idx] for idx in range(start, end)]
    source = torch.stack([item[0] for item in items])
    target = torch.stack([item[1] for item in items])
    source_meta = torch.stack([item[3] for item in items])
    target_meta = torch.stack([dataset.get_metadata(idx)["unit_ids_by_layer"][1:] for idx in range(start, end)])
    return source, target, source_meta, target_meta


def bucket_mask_for_targets(target_meta, bucket_ids, args, bucket_name=None):
    target_feature = target_meta[:, :, args.frequency_feature_layer]
    valid = target_feature.ge(0)
    safe_feature = target_feature.clamp_min(0)
    target_bucket = bucket_ids[safe_feature]
    if bucket_name is None:
        return valid
    bucket_idx = {"head": 0, "middle": 1, "tail": 2}[bucket_name]
    return valid & (target_bucket == bucket_idx)


def new_counter():
    return {"loss_sum": 0.0, "correct": 0, "count": 0}


def update_counter(counter, loss, pred, target, mask):
    if int(mask.sum().item()) == 0:
        return
    counter["loss_sum"] += float(loss[mask].sum().item())
    counter["correct"] += int((pred[mask] == target[mask]).sum().item())
    counter["count"] += int(mask.sum().item())


def finish_counter(counter):
    if counter["count"] == 0:
        return {"loss": None, "perplexity": None, "accuracy": None, "count": 0}
    loss = counter["loss_sum"] / counter["count"]
    return {
        "loss": loss,
        "perplexity": math.exp(min(loss, 50.0)),
        "accuracy": counter["correct"] / counter["count"],
        "count": counter["count"],
    }


@torch.no_grad()
def analyze_learning_curve(args, model, dataset, bucket_ids, device):
    counters = {"overall": new_counter()}
    for name in ["head", "middle", "tail"]:
        counters[name] = new_counter()
    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        source, target, _, target_meta = batch_from_dataset(dataset, start, end)
        source = source.to(device)
        target = target.to(device)
        target_meta = target_meta.to(device)
        logits = model(source).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
        loss = loss.reshape_as(target)
        pred = logits.argmax(dim=-1)
        valid = target.ne(151643)
        update_counter(counters["overall"], loss, pred, target, valid)
        for name in ["head", "middle", "tail"]:
            mask = valid & bucket_mask_for_targets(target_meta, bucket_ids.to(device), args, name)
            update_counter(counters[name], loss, pred, target, mask)
    return {name: finish_counter(counter) for name, counter in counters.items()}


def selected_parameters(model, regex_text):
    pattern = re.compile(regex_text) if regex_text else None
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if pattern is not None and not pattern.search(name):
            continue
        params.append((name, param))
    return params


def flatten_grads(params):
    chunks = []
    for _, param in params:
        if param.grad is None:
            chunks.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            chunks.append(param.grad.detach().float().cpu().reshape(-1))
    if not chunks:
        return torch.zeros(0)
    return torch.cat(chunks, dim=0)


def masked_loss(model, source, target, target_meta, bucket_ids, args, bucket_name, device):
    logits = model(source.to(device)).logits
    target = target.to(device)
    target_meta = target_meta.to(device)
    token_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none")
    token_loss = token_loss.reshape_as(target)
    valid = target.ne(151643)
    if bucket_name != "mix":
        valid = valid & bucket_mask_for_targets(target_meta, bucket_ids.to(device), args, bucket_name)
    if int(valid.sum().item()) == 0:
        return None
    return token_loss[valid].mean()


def grad_for_loss(model, params, loss):
    model.zero_grad(set_to_none=True)
    if loss is None:
        return torch.zeros(sum(param.numel() for _, param in params))
    loss.backward()
    grad = flatten_grads(params)
    model.zero_grad(set_to_none=True)
    return grad


def cosine(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return None
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return None
    return float(torch.dot(a, b).item() / denom)


def analyze_gradients(args, model, train_distribution, bucket_ids, device):
    params = selected_parameters(model, args.gradient_param_regex)
    mix_dataset = make_dataset(args, train_distribution, num_samples=args.gradient_samples)
    uniform_dataset = make_dataset(args, "uniform", num_samples=args.gradient_samples)

    source_mix, target_mix, _, target_meta_mix = batch_from_dataset(
        mix_dataset, 0, min(args.gradient_batch_size, len(mix_dataset))
    )
    g = {}
    g["mix"] = grad_for_loss(
        model,
        params,
        masked_loss(model, source_mix, target_mix, target_meta_mix, bucket_ids, args, "mix", device),
    )
    source, target, _, target_meta = batch_from_dataset(
        uniform_dataset, 0, min(args.gradient_batch_size, len(uniform_dataset))
    )
    for name in ["head", "middle", "tail"]:
        g[name] = grad_for_loss(
            model,
            params,
            masked_loss(model, source, target, target_meta, bucket_ids, args, name, device),
        )

    metrics = {
        "param_count": int(sum(param.numel() for _, param in params)),
        "norms": {name: float(vec.norm().item()) for name, vec in g.items()},
        "cos": {},
        "dot": {},
    }
    names = ["mix", "head", "middle", "tail"]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}__{b}"
            metrics["cos"][key] = cosine(g[a], g[b])
            metrics["dot"][key] = float(torch.dot(g[a], g[b]).item()) if g[a].numel() else None
    mix_head = metrics["cos"].get("mix__head")
    mix_tail = metrics["cos"].get("mix__tail")
    metrics["alignment_gap_head_minus_tail"] = (
        None if mix_head is None or mix_tail is None else float(mix_head - mix_tail)
    )
    metrics["head_descent_effect_on_tail"] = (
        None if g["head"].numel() == 0 else float(-torch.dot(g["head"], g["tail"]).item())
    )
    metrics["tail_descent_effect_on_head"] = metrics["head_descent_effect_on_tail"]
    return metrics


@torch.no_grad()
def collect_hidden(args, model, dataset, device, max_samples):
    rep_chunks = None
    high_chunks = []
    token_chunks = []
    target_high_chunks = []
    for start in range(0, min(len(dataset), max_samples), args.batch_size):
        end = min(start + args.batch_size, min(len(dataset), max_samples))
        source, target, _, target_meta = batch_from_dataset(dataset, start, end)
        output = model(source.to(device), output_hidden_states=True)
        hidden_states = output.hidden_states
        if rep_chunks is None:
            rep_chunks = [[] for _ in hidden_states]
        for idx, states in enumerate(hidden_states):
            rep_chunks[idx].append(states.detach().float().cpu().reshape(-1, states.shape[-1]))
        high_chunks.append(target_meta[:, :, args.frequency_feature_layer].reshape(-1))
        target_high_chunks.append(target_meta[:, :, args.frequency_feature_layer].reshape(-1))
        token_chunks.append(target.reshape(-1))
    reps = [torch.cat(chunks, dim=0) for chunks in rep_chunks]
    labels = {
        "higher_unit": torch.cat(high_chunks, dim=0),
        "target_higher_unit": torch.cat(target_high_chunks, dim=0),
        "target_token": torch.cat(token_chunks, dim=0),
    }
    return reps, labels


def effective_rank(eigenvalues):
    vals = eigenvalues.clamp_min(0)
    total = vals.sum()
    if float(total.item()) == 0.0:
        return 0.0
    probs = vals / total
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def feature_centroids(x, labels, min_count=1):
    sums = defaultdict(lambda: torch.zeros(x.shape[-1]))
    counts = Counter()
    for feature_id in torch.unique(labels).tolist():
        if feature_id < 0:
            continue
        mask = labels == feature_id
        count = int(mask.sum().item())
        if count < min_count:
            continue
        sums[int(feature_id)] = x[mask].mean(dim=0)
        counts[int(feature_id)] = count
    return sums, counts


def bucket_for_feature(feature_id, bucket_ids):
    return ["head", "middle", "tail"][int(bucket_ids[int(feature_id)].item())]


def analyze_svd(args, model, dataset, bucket_ids, device):
    reps, labels = collect_hidden(args, model, dataset, device, args.svd_samples)
    top_dims = parse_ints(args.svd_top_dims)
    results = []
    for layer_idx, x in enumerate(reps):
        x = x.float()
        x_centered = x - x.mean(dim=0, keepdim=True)
        _, s, vh = torch.linalg.svd(x_centered, full_matrices=False)
        eigen = s.pow(2) / max(x_centered.shape[0] - 1, 1)
        total = float(eigen.sum().item())
        pcs = vh
        centroids, counts = feature_centroids(x_centered, labels["higher_unit"])
        bucket_stats = {}
        for bucket in ["head", "middle", "tail"]:
            vectors = [
                vec for feature_id, vec in centroids.items()
                if bucket_for_feature(feature_id, bucket_ids) == bucket
            ]
            if not vectors:
                bucket_stats[bucket] = {"num_features": 0}
                continue
            mat = torch.stack(vectors)
            stats = {
                "num_features": len(vectors),
                "centroid_norm_mean": float(mat.norm(dim=-1).mean().item()),
            }
            for k in top_dims:
                kk = min(k, pcs.shape[0])
                proj = mat @ pcs[:kk].T
                denom = mat.pow(2).sum(dim=-1).clamp_min(1e-12)
                stats[f"top{k}_projection_energy_fraction"] = float((proj.pow(2).sum(dim=-1) / denom).mean().item())
            bucket_stats[bucket] = stats
        explained = {}
        running = torch.cumsum(eigen, dim=0)
        for k in top_dims:
            kk = min(k, eigen.numel())
            explained[f"top{k}"] = None if total == 0.0 else float((running[kk - 1] / total).item())
        results.append(
            {
                "layer": layer_idx,
                "hidden_dim": int(x.shape[-1]),
                "effective_rank": effective_rank(eigen),
                "explained_variance": explained,
                "bucket_stats": bucket_stats,
                "feature_count": int(len(counts)),
            }
        )
    return results


def centroid_probe(x, labels, bucket_ids, train_fraction, seed):
    valid = labels.ge(0)
    x = F.normalize(x[valid].float(), dim=-1)
    labels = labels[valid].long()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    train_len = int(x.shape[0] * train_fraction)
    train_idx, test_idx = perm[:train_len], perm[train_len:]
    train_x, train_y = x[train_idx], labels[train_idx]
    test_x, test_y = x[test_idx], labels[test_idx]
    centroids = []
    classes = []
    for cls in torch.unique(train_y).tolist():
        mask = train_y == cls
        if int(mask.sum().item()) == 0:
            continue
        centroids.append(F.normalize(train_x[mask].mean(dim=0), dim=0))
        classes.append(int(cls))
    if not centroids or test_x.shape[0] == 0:
        return {"overall_accuracy": None, "bucket_accuracy": {}}
    centroid_mat = torch.stack(centroids)
    class_tensor = torch.tensor(classes, dtype=torch.long)
    pred = class_tensor[(test_x @ centroid_mat.T).argmax(dim=-1)]
    overall = float((pred == test_y).float().mean().item())
    bucket_acc = {}
    for bucket in ["head", "middle", "tail"]:
        bucket_idx = {"head": 0, "middle": 1, "tail": 2}[bucket]
        mask = bucket_ids[test_y] == bucket_idx
        if int(mask.sum().item()) == 0:
            bucket_acc[bucket] = None
        else:
            bucket_acc[bucket] = float((pred[mask] == test_y[mask]).float().mean().item())
    majority = Counter(test_y.tolist()).most_common(1)[0][1] / max(int(test_y.numel()), 1)
    return {"overall_accuracy": overall, "majority_baseline": majority, "bucket_accuracy": bucket_acc}


def analyze_probe(args, model, dataset, bucket_ids, device):
    reps, labels = collect_hidden(args, model, dataset, device, args.svd_samples)
    results = []
    for layer_idx, x in enumerate(reps):
        results.append(
            {
                "layer": layer_idx,
                "higher_unit_centroid_probe": centroid_probe(
                    x,
                    labels["higher_unit"],
                    bucket_ids,
                    args.probe_train_fraction,
                    args.random_seed + layer_idx,
                ),
            }
        )
    return results


@torch.no_grad()
def analyze_lm_bias(args, model, dataset, bucket_ids, device):
    counters = {}
    for name in ["head", "middle", "tail"]:
        counters[name] = {
            "count": 0,
            "target_prob_sum": 0.0,
            "target_margin_sum": 0.0,
            "embed_norm_sum": 0.0,
            "lm_head_norm_sum": 0.0,
            "pred_head_count": 0,
        }
    embed_weight = model.model.embed_tokens.weight.detach().float().cpu()
    lm_head_weight = model.lm_head.weight.detach().float().cpu()
    token_bucket_counts = defaultdict(Counter)
    for idx in range(len(dataset)):
        _, target, _, target_meta = batch_from_dataset(dataset, idx, idx + 1)
        target_feature = target_meta[0, :, args.frequency_feature_layer]
        valid = target_feature.ge(0)
        safe_feature = target_feature.clamp_min(0)
        target_bucket = bucket_ids[safe_feature]
        for token_id, bucket_idx in zip(target[0, valid].tolist(), target_bucket[valid].tolist()):
            token_bucket_counts[int(token_id)][int(bucket_idx)] += 1
    token_to_bucket = {
        token_id: counts.most_common(1)[0][0]
        for token_id, counts in token_bucket_counts.items()
        if counts
    }

    for start in range(0, len(dataset), args.batch_size):
        end = min(start + args.batch_size, len(dataset))
        source, target, _, target_meta = batch_from_dataset(dataset, start, end)
        logits = model(source.to(device)).logits.detach().float().cpu()
        probs = F.softmax(logits, dim=-1)
        top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
        pred = top2.indices[..., 0]
        target_feature = target_meta[:, :, args.frequency_feature_layer]
        valid = target_feature.ge(0)
        safe_feature = target_feature.clamp_min(0)
        target_bucket = bucket_ids[safe_feature]
        for bucket_name, bucket_idx in [("head", 0), ("middle", 1), ("tail", 2)]:
            mask = valid & (target_bucket == bucket_idx)
            if int(mask.sum().item()) == 0:
                continue
            target_ids = target[mask]
            target_logits = logits[mask, :].gather(1, target_ids[:, None]).squeeze(1)
            masked_top_values = top2.values[mask]
            masked_top_indices = top2.indices[mask]
            best_logits = masked_top_values[:, 0]
            second_logits = masked_top_values[:, 1] if masked_top_values.shape[-1] > 1 else best_logits
            best_ids = masked_top_indices[:, 0]
            margin = torch.where(best_ids == target_ids, target_logits - second_logits, target_logits - best_logits)
            target_probs = probs[mask, :].gather(1, target_ids[:, None]).squeeze(1)
            counter = counters[bucket_name]
            counter["count"] += int(mask.sum().item())
            counter["target_prob_sum"] += float(target_probs.sum().item())
            counter["target_margin_sum"] += float(margin.sum().item())
            counter["embed_norm_sum"] += float(embed_weight[target_ids].norm(dim=-1).sum().item())
            counter["lm_head_norm_sum"] += float(lm_head_weight[target_ids].norm(dim=-1).sum().item())
            pred_ids = pred[mask]
            pred_head = 0
            for pred_id in pred_ids.tolist():
                if token_to_bucket.get(int(pred_id)) == 0:
                    pred_head += 1
            counter["pred_head_count"] += pred_head

    finished = {}
    for bucket_name, counter in counters.items():
        count = max(counter["count"], 1)
        finished[bucket_name] = {
            "count": counter["count"],
            "target_prob_mean": counter["target_prob_sum"] / count,
            "target_margin_mean": counter["target_margin_sum"] / count,
            "target_embed_norm_mean": counter["embed_norm_sum"] / count,
            "target_lm_head_norm_mean": counter["lm_head_norm_sum"] / count,
            "pred_head_rate_approx": counter["pred_head_count"] / count,
        }
    return finished


def main():
    args = parse_args()
    modes = parse_modes(args.modes)
    run_specs = parse_run_specs(args.run_specs)
    device = choose_device()
    results = {"config": vars(args), "device": str(device), "runs": []}

    for spec in run_specs:
        run_dir = os.path.join(args.checkpoint_root, spec["run_name"])
        checkpoints = list_checkpoints(run_dir, args.checkpoint_steps)
        bucket_ids, bucket_meta = make_bucket_ids(args, spec["train_distribution"])
        eval_dataset = make_dataset(args, args.eval_sampling_distribution, num_samples=args.num_samples)
        run_result = {
            **spec,
            "bucket_meta": bucket_meta,
            "checkpoints": [],
        }
        for step, checkpoint_path in checkpoints:
            print(f"analyze {spec['alias']} step={step}", flush=True)
            model, config = load_model(args, run_dir, checkpoint_path, device)
            row = {
                "step": step,
                "checkpoint": checkpoint_path,
                "model": {
                    "hidden_size": int(config.hidden_size),
                    "intermediate_size": int(config.intermediate_size),
                    "num_hidden_layers": int(config.num_hidden_layers),
                    "num_attention_heads": int(config.num_attention_heads),
                    "head_dim": int(config.head_dim),
                    "use_moe": bool(getattr(config, "use_moe", False)),
                },
            }
            if "learning_curve" in modes:
                row["learning_curve"] = analyze_learning_curve(args, model, eval_dataset, bucket_ids, device)
            if "gradients" in modes:
                row["gradients"] = analyze_gradients(args, model, spec["train_distribution"], bucket_ids, device)
            if "svd" in modes:
                row["svd"] = analyze_svd(args, model, eval_dataset, bucket_ids, device)
            if "probe" in modes:
                row["probe"] = analyze_probe(args, model, eval_dataset, bucket_ids, device)
            if "lm_bias" in modes:
                row["lm_bias"] = analyze_lm_bias(args, model, eval_dataset, bucket_ids, device)
            run_result["checkpoints"].append(row)
            del model
        results["runs"].append(run_result)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
